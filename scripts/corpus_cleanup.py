#!/usr/bin/env python3
"""
Corpus cleanup: fix metadata gaps and text artifacts before production.

Tasks handled (all local, no API calls):
  1. Strip markdown bold (**text**) from olmOCR extractions
  2. Backfill DB columns from JSON: page_count, word_count, letter_date, requestor_name
  3. Generate display titles for bare "Year: YYYY Advice Letter #" entries
  4. Decode HTML entities in titles

Usage:
    python scripts/corpus_cleanup.py --all          # Run all cleanup tasks
    python scripts/corpus_cleanup.py --strip-bold   # Just strip markdown bold
    python scripts/corpus_cleanup.py --backfill     # Just backfill DB from JSON
    python scripts/corpus_cleanup.py --titles       # Just fix titles
    python scripts/corpus_cleanup.py --dry-run      # Preview changes without saving
"""

import argparse
import html
import json
import os
import re
import sys
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from scraper.config import DATA_DIR
from scraper.db import get_connection
from scraper.quality import compute_quality_score


# ---------------------------------------------------------------------------
# Task 1: Strip markdown bold from olmOCR docs
# ---------------------------------------------------------------------------

def strip_markdown_bold(dry_run: bool = False) -> dict:
    """Remove **bold** markdown from olmOCR extracted text in JSON files."""
    conn = get_connection()
    cursor = conn.cursor()

    cursor.execute("""
        SELECT id, json_path, extraction_quality, page_count
        FROM documents
        WHERE extraction_method = 'olmocr'
    """)
    docs = [dict(row) for row in cursor.fetchall()]

    bold_pattern = re.compile(r'\*\*(.+?)\*\*')
    modified = 0
    skipped = 0

    for doc in docs:
        json_path = doc["json_path"]
        if not json_path:
            continue
        full_path = os.path.join(str(DATA_DIR.parent), json_path)
        if not os.path.exists(full_path):
            continue

        with open(full_path) as f:
            data = json.load(f)

        full_text = data.get("content", {}).get("full_text", "")
        if "**" not in full_text:
            skipped += 1
            continue

        cleaned = bold_pattern.sub(r'\1', full_text)
        if cleaned == full_text:
            skipped += 1
            continue

        modified += 1

        if dry_run:
            # Show a sample
            if modified <= 5:
                # Find first bold occurrence for preview
                match = bold_pattern.search(full_text)
                if match:
                    start = max(0, match.start() - 30)
                    end = min(len(full_text), match.end() + 30)
                    print(f"  [{doc['id']}] ...{full_text[start:end]}...")
            continue

        # Update JSON file
        data["content"]["full_text"] = cleaned
        if data["content"].get("full_text_markdown"):
            data["content"]["full_text_markdown"] = bold_pattern.sub(
                r'\1', data["content"]["full_text_markdown"]
            )

        # Rescore quality
        page_count = data.get("extraction", {}).get("page_count") or doc.get("page_count") or 1
        new_metrics = compute_quality_score(cleaned, page_count)
        data["extraction"]["quality_score"] = new_metrics.final_score

        with open(full_path, "w") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)

        # Update DB quality score
        cursor.execute(
            "UPDATE documents SET extraction_quality = ? WHERE id = ?",
            (new_metrics.final_score, doc["id"])
        )

    if not dry_run:
        conn.commit()
    conn.close()

    return {"modified": modified, "skipped": skipped, "total": len(docs)}


# ---------------------------------------------------------------------------
# Task 2: Backfill DB from JSON files
# ---------------------------------------------------------------------------

def backfill_db_from_json(dry_run: bool = False) -> dict:
    """Backfill page_count, word_count, letter_date, requestor_name from JSON."""
    conn = get_connection()
    cursor = conn.cursor()

    cursor.execute("""
        SELECT id, json_path, page_count, word_count, letter_date, requestor_name
        FROM documents
    """)
    docs = [dict(row) for row in cursor.fetchall()]

    stats = {
        "page_count_filled": 0,
        "word_count_filled": 0,
        "letter_date_filled": 0,
        "requestor_name_filled": 0,
        "errors": 0,
    }

    for doc in docs:
        json_path = doc["json_path"]
        if not json_path:
            continue
        full_path = os.path.join(str(DATA_DIR.parent), json_path)
        if not os.path.exists(full_path):
            continue

        try:
            with open(full_path) as f:
                data = json.load(f)
        except (json.JSONDecodeError, OSError):
            stats["errors"] += 1
            continue

        extraction = data.get("extraction", {})
        parsed = data.get("parsed", {})
        updates = {}

        # page_count
        if not doc["page_count"] and extraction.get("page_count"):
            updates["page_count"] = extraction["page_count"]
            stats["page_count_filled"] += 1

        # word_count
        if not doc["word_count"] and extraction.get("word_count"):
            updates["word_count"] = extraction["word_count"]
            stats["word_count_filled"] += 1

        # letter_date — prefer ISO format from parsed.date
        if not doc["letter_date"] and parsed.get("date"):
            updates["letter_date"] = parsed["date"]
            stats["letter_date_filled"] += 1

        # requestor_name
        if not doc["requestor_name"] and parsed.get("requestor_name"):
            updates["requestor_name"] = parsed["requestor_name"]
            stats["requestor_name_filled"] += 1

        if updates and not dry_run:
            set_clause = ", ".join(f"{k} = ?" for k in updates)
            values = list(updates.values()) + [doc["id"]]
            cursor.execute(f"UPDATE documents SET {set_clause} WHERE id = ?", values)

    if not dry_run:
        conn.commit()
    conn.close()

    stats["total_docs"] = len(docs)
    return stats


# ---------------------------------------------------------------------------
# Task 3: Generate display titles
# ---------------------------------------------------------------------------

BARE_TITLE_PATTERN = re.compile(
    r'^Year:\s*\d{4}\s+Advice\s+Letter\s*#?\s*(.*)$',
    re.IGNORECASE
)


def fix_titles(dry_run: bool = False) -> dict:
    """
    Generate descriptive display titles for bare entries and decode HTML entities.
    """
    conn = get_connection()
    cursor = conn.cursor()

    cursor.execute("""
        SELECT id, title_text, letter_id, requestor_name, letter_date, city, year_tag, json_path
        FROM documents
    """)
    docs = [dict(row) for row in cursor.fetchall()]

    stats = {
        "titles_enriched": 0,
        "html_decoded": 0,
        "already_good": 0,
    }

    for doc in docs:
        title = doc["title_text"] or ""
        new_title = title
        changed = False

        # Decode HTML entities first (applies to all titles)
        if "&#" in new_title or "&amp;" in new_title or "&lt;" in new_title or "&gt;" in new_title or "&quot;" in new_title:
            decoded = html.unescape(new_title)
            if decoded != new_title:
                new_title = decoded
                changed = True
                stats["html_decoded"] += 1

        # Check for bare "Year: YYYY Advice Letter # NNN" pattern
        match = BARE_TITLE_PATTERN.match(new_title)
        if match:
            # Try to build a better title from available metadata
            parts = []

            # Get requestor name — try DB first, then JSON
            name = doc.get("requestor_name")
            if not name and doc.get("json_path"):
                full_path = os.path.join(str(DATA_DIR.parent), doc["json_path"])
                if os.path.exists(full_path):
                    try:
                        with open(full_path) as f:
                            data = json.load(f)
                        name = data.get("parsed", {}).get("requestor_name")
                    except (json.JSONDecodeError, OSError):
                        pass

            if name:
                parts.append(name)

            # Letter ID
            lid = doc.get("letter_id")
            if lid:
                parts.append(lid)

            # Date
            date = doc.get("letter_date")
            if not date and doc.get("json_path"):
                full_path = os.path.join(str(DATA_DIR.parent), doc["json_path"])
                if os.path.exists(full_path):
                    try:
                        with open(full_path) as f:
                            data = json.load(f)
                        date = data.get("parsed", {}).get("date")
                    except (json.JSONDecodeError, OSError):
                        pass

            if date:
                parts.append(date)

            # City
            city = doc.get("city")
            if city:
                parts.append(city)

            if parts:
                new_title = " - ".join(parts)
                changed = True
                stats["titles_enriched"] += 1
            # If we have no metadata at all, keep the bare title
        else:
            if not changed:
                stats["already_good"] += 1

        if changed and not dry_run:
            cursor.execute(
                "UPDATE documents SET title_text = ? WHERE id = ?",
                (new_title, doc["id"])
            )
        elif changed and dry_run and (stats["titles_enriched"] + stats["html_decoded"]) <= 10:
            print(f"  [{doc['id']}] {title[:60]}")
            print(f"       → {new_title[:60]}")

    if not dry_run:
        conn.commit()
    conn.close()

    stats["total_docs"] = len(docs)
    return stats


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Corpus cleanup: fix metadata and text artifacts"
    )
    parser.add_argument("--all", action="store_true",
                        help="Run all cleanup tasks")
    parser.add_argument("--strip-bold", action="store_true",
                        help="Strip markdown bold from olmOCR docs")
    parser.add_argument("--backfill", action="store_true",
                        help="Backfill DB from JSON files")
    parser.add_argument("--titles", action="store_true",
                        help="Fix titles (enrich bare entries + decode HTML)")
    parser.add_argument("--dry-run", action="store_true",
                        help="Preview changes without saving")
    args = parser.parse_args()

    if not (args.all or args.strip_bold or args.backfill or args.titles):
        parser.print_help()
        sys.exit(1)

    if args.dry_run:
        print("DRY RUN — no changes will be saved\n")

    # Task 1: Strip markdown bold
    if args.all or args.strip_bold:
        print("=" * 60)
        print("TASK: Strip markdown bold from olmOCR docs")
        print("=" * 60)
        result = strip_markdown_bold(dry_run=args.dry_run)
        print(f"  Total olmOCR docs: {result['total']}")
        print(f"  Modified: {result['modified']}")
        print(f"  Skipped (no bold): {result['skipped']}")
        print()

    # Task 2: Backfill DB from JSON
    if args.all or args.backfill:
        print("=" * 60)
        print("TASK: Backfill DB from JSON files")
        print("=" * 60)
        result = backfill_db_from_json(dry_run=args.dry_run)
        print(f"  Total docs: {result['total_docs']}")
        print(f"  page_count filled: {result['page_count_filled']}")
        print(f"  word_count filled: {result['word_count_filled']}")
        print(f"  letter_date filled: {result['letter_date_filled']}")
        print(f"  requestor_name filled: {result['requestor_name_filled']}")
        if result['errors']:
            print(f"  JSON read errors: {result['errors']}")
        print()

    # Task 3+4: Fix titles (enrich + HTML decode)
    if args.all or args.titles:
        print("=" * 60)
        print("TASK: Fix titles (enrich bare entries + decode HTML)")
        print("=" * 60)
        result = fix_titles(dry_run=args.dry_run)
        print(f"  Total docs: {result['total_docs']}")
        print(f"  Titles enriched: {result['titles_enriched']}")
        print(f"  HTML entities decoded: {result['html_decoded']}")
        print(f"  Already good: {result['already_good']}")
        print()

    print("Done.")


if __name__ == "__main__":
    main()
