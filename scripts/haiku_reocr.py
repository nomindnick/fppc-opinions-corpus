#!/usr/bin/env python3
"""
Re-extract flagged documents using Claude Haiku as a vision OCR engine.

For documents where the Tesseract canary flagged fidelity issues (critical, high,
medium risk tiers), this script uses Claude Haiku to re-OCR every page directly.
Unlike the verify_high_risk.py approach (which only checked page 1 and fell back
to Tesseract), this does a full multi-page transcription via Haiku — producing
text that is both faithful AND readable.

The script operates in two modes:
  --sample N   Run a targeted sample of N docs (spread across tiers/years) to
               validate the approach before committing to the full batch.
  --all        Process all flagged documents.

Requires: ANTHROPIC_API_KEY

Usage:
    python scripts/haiku_reocr.py --sample 15           # Sample test (recommended first)
    python scripts/haiku_reocr.py --sample 15 --dry-run # Preview sample selection
    python scripts/haiku_reocr.py --all --dry-run       # Preview full batch
    python scripts/haiku_reocr.py --all                  # Full batch
    python scripts/haiku_reocr.py --tier critical        # Only critical tier
    python scripts/haiku_reocr.py --tier high            # Only high tier
    python scripts/haiku_reocr.py --doc-id 5252          # Single document by DB id
"""

import argparse
import base64
import json
import os
import random
import sys
import time
from datetime import datetime, timezone
from difflib import SequenceMatcher
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dotenv import load_dotenv
load_dotenv()

import anthropic

from scraper.config import DATA_DIR, RAW_PDFS_DIR
from scraper.db import get_connection, update_extraction_status, update_fidelity
from scraper.quality import compute_quality_score

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

HAIKU_MODEL = "claude-haiku-4-5-20251001"

# Haiku 4.5 pricing (per million tokens)
HAIKU_INPUT_COST = 0.80
HAIKU_OUTPUT_COST = 4.00

REPORT_PATH = DATA_DIR / "qa_reports" / "haiku_reocr_report.json"
SAMPLE_REPORT_PATH = DATA_DIR / "qa_reports" / "haiku_reocr_sample.json"

# olmOCR docs with LLM chatbot artifacts (identified by hallucination audit)
CONTAMINATED_IDS = [
    342, 536, 1333, 1490, 4628, 4630, 4692, 4703, 4717, 4721, 4739, 4769,
    4808, 4819, 4835, 4842, 4856, 4859, 4890, 4896, 4902, 4908, 4912, 4915,
    4939, 4965, 5029, 5033, 5060, 8884, 8944, 8969, 9033, 9052, 9100, 9154,
    9172, 9193, 9203, 9242, 9280, 9287, 9300, 9306, 9387, 9429, 9457, 9577,
    9985, 10092, 10248, 10352, 10430, 10446, 10455, 10498, 10523, 10560,
    10563, 10570, 10580, 10601, 10604, 10617, 10620, 10634, 10682, 11049,
    11124, 11129, 11182, 11228, 11230, 11240, 11246, 11259, 11304, 11341,
    11393, 11419, 11434, 11443, 11449, 11470, 11486,
]

# Transcription prompt — strict, no room for description-mode or paraphrasing
TRANSCRIPTION_PROMPT = """Transcribe exactly what you see in this document image.
Rules:
- Copy the text verbatim, preserving the original wording, spelling, and punctuation.
- Maintain paragraph breaks.
- Do NOT summarize, paraphrase, or describe the document.
- Do NOT add commentary, headers, or labels that are not in the original.
- If a word or section is illegible, write [illegible] in its place.
- Return ONLY the transcribed text."""

# Rate limit: stay well under Haiku tier limits
REQUEST_DELAY = 0.3  # seconds between API calls


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def resolve_pdf_path(doc_row: dict) -> str | None:
    """Resolve local PDF path from DB row."""
    year = doc_row.get("year_tag")
    pdf_url = doc_row.get("pdf_url", "")
    filename = pdf_url.rstrip("/").split("/")[-1]
    if not filename.lower().endswith(".pdf"):
        filename += ".pdf"

    year_dir = RAW_PDFS_DIR / str(year)
    pdf_path = year_dir / filename

    if pdf_path.exists():
        return str(pdf_path)

    # Case-insensitive fallback
    if year_dir.is_dir():
        target_stem = Path(filename).stem.lower()
        for candidate in year_dir.iterdir():
            if candidate.stem.lower() == target_stem and candidate.suffix.lower() == ".pdf":
                return str(candidate)
    return None


def render_page_to_png(pdf_path: str, page_num: int = 0, dpi: int = 200) -> bytes:
    """Render a single PDF page to PNG bytes."""
    import fitz

    doc = fitz.open(pdf_path)
    try:
        page = doc[page_num]
        mat = fitz.Matrix(dpi / 72, dpi / 72)
        pix = page.get_pixmap(matrix=mat)
        return pix.tobytes("png")
    finally:
        doc.close()


def get_page_count(pdf_path: str) -> int:
    """Return the number of pages in a PDF."""
    import fitz
    doc = fitz.open(pdf_path)
    count = len(doc)
    doc.close()
    return count


def normalize_for_comparison(text: str) -> str:
    """Normalize text for word-level comparison."""
    import re
    text = text.lower()
    text = re.sub(r'[^\w\s]', ' ', text)
    text = re.sub(r'\s+', ' ', text)
    return text.strip()


def get_existing_text(json_path: str) -> str:
    """Load the current full_text from a document JSON."""
    with open(json_path) as f:
        data = json.load(f)
    return data.get("content", {}).get("full_text", "")


MAX_IMAGE_B64_BYTES = 5 * 1024 * 1024  # Anthropic's 5MB limit on base64-encoded images


def _b64_size(raw_bytes: bytes) -> int:
    """Estimate base64-encoded size without actually encoding (raw * 4/3, rounded up)."""
    return ((len(raw_bytes) + 2) // 3) * 4


def transcribe_page(
    client: anthropic.Anthropic,
    pdf_path: str,
    page_num: int,
    dpi: int = 200,
) -> dict:
    """
    Transcribe a single PDF page using Haiku vision.

    If the rendered image exceeds 5MB after base64 encoding, automatically
    retries at lower DPI.
    Returns dict with: text, input_tokens, output_tokens, cost
    """
    img_bytes = render_page_to_png(pdf_path, page_num=page_num, dpi=dpi)

    # Downscale if base64-encoded image would exceed API limit
    if _b64_size(img_bytes) > MAX_IMAGE_B64_BYTES:
        for fallback_dpi in (150, 120, 100):
            img_bytes = render_page_to_png(pdf_path, page_num=page_num, dpi=fallback_dpi)
            if _b64_size(img_bytes) <= MAX_IMAGE_B64_BYTES:
                break
        else:
            return {"text": "", "input_tokens": 0, "output_tokens": 0, "cost": 0.0,
                    "error": f"page image too large even at 100 DPI ({_b64_size(img_bytes)} bytes b64)"}

    img_b64 = base64.b64encode(img_bytes).decode("utf-8")

    for attempt in range(3):
        try:
            response = client.messages.create(
                model=HAIKU_MODEL,
                max_tokens=4096,
                messages=[{
                    "role": "user",
                    "content": [
                        {
                            "type": "image",
                            "source": {
                                "type": "base64",
                                "media_type": "image/png",
                                "data": img_b64,
                            },
                        },
                        {
                            "type": "text",
                            "text": TRANSCRIPTION_PROMPT,
                        },
                    ],
                }],
            )

            text = response.content[0].text
            input_tokens = response.usage.input_tokens
            output_tokens = response.usage.output_tokens
            cost = (input_tokens / 1_000_000 * HAIKU_INPUT_COST
                    + output_tokens / 1_000_000 * HAIKU_OUTPUT_COST)

            return {
                "text": text,
                "input_tokens": input_tokens,
                "output_tokens": output_tokens,
                "cost": cost,
            }

        except anthropic.RateLimitError:
            wait = 2 ** (attempt + 2)  # 4, 8, 16 seconds
            print(f"    Rate limited, waiting {wait}s...")
            time.sleep(wait)
        except anthropic.APIStatusError as e:
            if e.status_code >= 500:
                time.sleep(2 ** (attempt + 1))
            else:
                raise

    return {"text": "", "input_tokens": 0, "output_tokens": 0, "cost": 0.0, "error": "retries exhausted"}


def transcribe_document(
    client: anthropic.Anthropic,
    pdf_path: str,
    max_pages: int = 20,
) -> dict:
    """
    Transcribe all pages of a PDF using Haiku vision.

    Returns dict with: text, page_count, total_input_tokens, total_output_tokens, total_cost, page_results
    """
    num_pages = min(get_page_count(pdf_path), max_pages)
    page_texts = []
    total_input = 0
    total_output = 0
    total_cost = 0.0
    page_results = []

    for page_num in range(num_pages):
        result = transcribe_page(client, pdf_path, page_num)

        if result.get("error"):
            page_results.append({"page": page_num, "error": result["error"]})
            continue

        page_texts.append(result["text"])
        total_input += result["input_tokens"]
        total_output += result["output_tokens"]
        total_cost += result["cost"]
        page_results.append({
            "page": page_num,
            "chars": len(result["text"]),
            "cost": round(result["cost"], 5),
        })

        # Rate limit delay between pages
        if page_num < num_pages - 1:
            time.sleep(REQUEST_DELAY)

    full_text = "\n\n".join(page_texts)

    return {
        "text": full_text,
        "page_count": num_pages,
        "total_input_tokens": total_input,
        "total_output_tokens": total_output,
        "total_cost": total_cost,
        "page_results": page_results,
    }


# ---------------------------------------------------------------------------
# Document selection
# ---------------------------------------------------------------------------

def get_flagged_docs(tier: str | None = None) -> list[dict]:
    """
    Get all flagged documents from the database.

    Args:
        tier: Optional filter — 'critical', 'high', or 'medium'. None = all.

    Returns:
        List of dicts with doc info.
    """
    conn = get_connection()
    cursor = conn.cursor()

    if tier:
        tiers = [tier]
    else:
        tiers = ["critical", "high", "medium"]

    placeholders = ",".join("?" for _ in tiers)
    cursor.execute(f"""
        SELECT id, letter_id, year_tag, extraction_method, extraction_quality,
               fidelity_score, fidelity_risk, json_path, pdf_url, page_count
        FROM documents
        WHERE fidelity_risk IN ({placeholders})
        ORDER BY fidelity_risk, year_tag
    """, tiers)

    docs = [dict(row) for row in cursor.fetchall()]
    conn.close()

    # Fill in missing page_count from JSON extraction metadata
    for d in docs:
        if not d.get("page_count") and d.get("json_path"):
            jp = os.path.join(str(DATA_DIR.parent), d["json_path"])
            if os.path.exists(jp):
                try:
                    with open(jp) as f:
                        data = json.load(f)
                    d["page_count"] = data.get("extraction", {}).get("page_count")
                except (json.JSONDecodeError, OSError):
                    pass

    return docs


def get_contaminated_docs() -> list[dict]:
    """Get olmOCR docs with LLM chatbot artifacts, excluding already re-extracted."""
    conn = get_connection()
    cursor = conn.cursor()
    placeholders = ",".join("?" for _ in CONTAMINATED_IDS)
    cursor.execute(f"""
        SELECT id, letter_id, year_tag, extraction_method, extraction_quality,
               fidelity_score, fidelity_risk, json_path, pdf_url, page_count
        FROM documents
        WHERE id IN ({placeholders})
          AND extraction_method != 'haiku_vision'
        ORDER BY year_tag
    """, CONTAMINATED_IDS)
    docs = [dict(row) for row in cursor.fetchall()]
    conn.close()

    # Fill missing page_count from JSON
    for d in docs:
        if not d.get("page_count") and d.get("json_path"):
            jp = os.path.join(str(DATA_DIR.parent), d["json_path"])
            if os.path.exists(jp):
                try:
                    with open(jp) as f:
                        data = json.load(f)
                    d["page_count"] = data.get("extraction", {}).get("page_count")
                except (json.JSONDecodeError, OSError):
                    pass
    return docs


def get_tesseract_docs() -> list[dict]:
    """Get all tesseract_fallback docs not yet re-extracted."""
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute("""
        SELECT id, letter_id, year_tag, extraction_method, extraction_quality,
               fidelity_score, fidelity_risk, json_path, pdf_url, page_count
        FROM documents
        WHERE extraction_method = 'tesseract_fallback'
        ORDER BY year_tag
    """)
    docs = [dict(row) for row in cursor.fetchall()]
    conn.close()

    # Fill missing page_count from JSON
    for d in docs:
        if not d.get("page_count") and d.get("json_path"):
            jp = os.path.join(str(DATA_DIR.parent), d["json_path"])
            if os.path.exists(jp):
                try:
                    with open(jp) as f:
                        data = json.load(f)
                    d["page_count"] = data.get("extraction", {}).get("page_count")
                except (json.JSONDecodeError, OSError):
                    pass
    return docs


def select_sample(docs: list[dict], n: int) -> list[dict]:
    """
    Select a stratified sample across risk tiers and year decades.

    Ensures representation from each tier and era. Targets:
    - At least 2 critical docs (if available)
    - Proportional representation of high and medium
    - Spread across year decades (1970s, 1980s, 1990s, 2000s+)
    """
    by_tier = {}
    for d in docs:
        tier = d["fidelity_risk"]
        by_tier.setdefault(tier, []).append(d)

    sample = []

    # Allocate slots per tier
    tier_counts = {}
    critical_count = min(len(by_tier.get("critical", [])), max(2, n // 5))
    remaining = n - critical_count
    high_docs = by_tier.get("high", [])
    medium_docs = by_tier.get("medium", [])
    total_hm = len(high_docs) + len(medium_docs)

    if total_hm > 0:
        high_count = min(len(high_docs), max(1, int(remaining * len(high_docs) / total_hm)))
        medium_count = min(len(medium_docs), remaining - high_count)
    else:
        high_count = 0
        medium_count = 0

    tier_counts = {"critical": critical_count, "high": high_count, "medium": medium_count}

    for tier, count in tier_counts.items():
        tier_docs = by_tier.get(tier, [])
        if not tier_docs or count == 0:
            continue

        # Stratify within tier by decade
        by_decade = {}
        for d in tier_docs:
            decade = (d["year_tag"] // 10) * 10
            by_decade.setdefault(decade, []).append(d)

        # Round-robin from each decade
        per_decade = max(1, count // len(by_decade))
        picked = []
        for decade in sorted(by_decade.keys()):
            candidates = by_decade[decade]
            random.shuffle(candidates)
            picked.extend(candidates[:per_decade])

        # Fill remaining slots randomly
        remaining_candidates = [d for d in tier_docs if d not in picked]
        random.shuffle(remaining_candidates)
        while len(picked) < count and remaining_candidates:
            picked.append(remaining_candidates.pop())

        sample.extend(picked[:count])

    return sample


# ---------------------------------------------------------------------------
# Core processing
# ---------------------------------------------------------------------------

def process_document(
    client: anthropic.Anthropic,
    doc: dict,
    update_db: bool = True,
) -> dict:
    """
    Re-extract a single document using Haiku vision OCR.

    Returns a result dict with before/after metrics.
    """
    doc_id = doc["id"]
    letter_id = doc["letter_id"] or f"doc#{doc_id}"
    json_path = doc.get("json_path")

    # Resolve PDF
    pdf_path = resolve_pdf_path(doc)
    if not pdf_path:
        return {"doc_id": doc_id, "letter_id": letter_id, "error": "PDF not found"}

    # Get existing text for comparison
    old_text = ""
    full_json_path = None
    if json_path:
        full_json_path = os.path.join(str(DATA_DIR.parent), json_path)
        if os.path.exists(full_json_path):
            old_text = get_existing_text(full_json_path)

    # Transcribe with Haiku
    haiku_result = transcribe_document(client, pdf_path)

    if not haiku_result["text"].strip():
        return {
            "doc_id": doc_id,
            "letter_id": letter_id,
            "error": "Haiku returned empty text",
            "cost": haiku_result["total_cost"],
        }

    new_text = haiku_result["text"]

    # Quality scores
    page_count = haiku_result["page_count"]
    old_metrics = compute_quality_score(old_text, page_count) if old_text else None
    new_metrics = compute_quality_score(new_text, page_count)

    # Fidelity comparison: Haiku vs old olmOCR text (word-level)
    if old_text:
        old_norm = normalize_for_comparison(old_text)
        new_norm = normalize_for_comparison(new_text)
        old_words = old_norm.split()
        new_words = new_norm.split()
        similarity = SequenceMatcher(None, old_words, new_words).ratio()
    else:
        similarity = None

    result = {
        "doc_id": doc_id,
        "letter_id": letter_id,
        "year": doc["year_tag"],
        "tier": doc["fidelity_risk"],
        "pages": page_count,
        "old_quality": round(old_metrics.final_score, 4) if old_metrics else None,
        "new_quality": round(new_metrics.final_score, 4),
        "quality_delta": round(new_metrics.final_score - old_metrics.final_score, 4) if old_metrics else None,
        "old_vs_new_similarity": round(similarity, 4) if similarity is not None else None,
        "old_word_count": len(old_text.split()) if old_text else 0,
        "new_word_count": len(new_text.split()),
        "cost": round(haiku_result["total_cost"], 5),
        "input_tokens": haiku_result["total_input_tokens"],
        "output_tokens": haiku_result["total_output_tokens"],
    }

    # Save updated JSON and DB
    if update_db and full_json_path and os.path.exists(full_json_path):
        with open(full_json_path) as f:
            doc_json = json.load(f)

        now = datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")

        doc_json["content"]["full_text"] = new_text
        doc_json["content"]["full_text_markdown"] = None
        doc_json["extraction"]["method"] = "haiku_vision"
        doc_json["extraction"]["quality_score"] = new_metrics.final_score
        doc_json["extraction"]["word_count"] = len(new_text.split())
        doc_json["extraction"]["char_count"] = len(new_text)
        doc_json["extraction"]["extracted_at"] = now
        doc_json["extraction"]["haiku_cost"] = haiku_result["total_cost"]

        with open(full_json_path, "w") as f:
            json.dump(doc_json, f, indent=2, ensure_ascii=False)

        update_extraction_status(
            doc_id=doc_id,
            status="extracted",
            method="haiku_vision",
            quality=new_metrics.final_score,
        )
        update_fidelity(doc_id, 0.95, "haiku_vision", "verified")

        result["saved"] = True
    else:
        result["saved"] = False

    return result


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Re-extract flagged documents using Claude Haiku vision OCR"
    )
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--sample", type=int, metavar="N",
                       help="Run targeted sample of N docs to validate approach")
    group.add_argument("--all", action="store_true",
                       help="Process all flagged documents")
    group.add_argument("--doc-id", type=int, metavar="ID",
                       help="Process a single document by DB id")
    group.add_argument("--contaminated", action="store_true",
                       help="Re-extract 85 olmOCR docs with LLM chatbot artifacts")
    group.add_argument("--tesseract", action="store_true",
                       help="Re-extract all tesseract_fallback docs")

    parser.add_argument("--tier", choices=["critical", "high", "medium"],
                        help="Filter to a specific risk tier")
    parser.add_argument("--dry-run", action="store_true",
                        help="Preview document selection without processing")
    parser.add_argument("--no-db-update", action="store_true",
                        help="Don't update JSON files or database")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for sample selection (default: 42)")
    args = parser.parse_args()

    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key and not args.dry_run:
        print("Error: ANTHROPIC_API_KEY not set")
        sys.exit(1)

    random.seed(args.seed)

    # --- Select documents ---
    mode_label = "flagged"
    if args.doc_id:
        conn = get_connection()
        cursor = conn.cursor()
        cursor.execute("SELECT * FROM documents WHERE id = ?", (args.doc_id,))
        row = cursor.fetchone()
        conn.close()
        if not row:
            print(f"Error: document {args.doc_id} not found")
            sys.exit(1)
        docs = [dict(row)]
        mode_label = "single"
    elif args.contaminated:
        docs = get_contaminated_docs()
        mode_label = "contaminated olmOCR"
    elif args.tesseract:
        docs = get_tesseract_docs()
        mode_label = "tesseract_fallback"
    else:
        docs = get_flagged_docs(tier=args.tier)

    print(f"Found {len(docs)} {mode_label} documents", end="")
    if args.tier:
        print(f" (tier={args.tier})", end="")
    print()

    # Tier breakdown
    tier_counts = {}
    total_pages = 0
    for d in docs:
        tier = d.get("fidelity_risk", "unknown")
        tier_counts[tier] = tier_counts.get(tier, 0) + 1
        total_pages += d.get("page_count", 0) or 0

    for tier in ["critical", "high", "medium"]:
        if tier in tier_counts:
            print(f"  {tier}: {tier_counts[tier]}")

    if args.sample:
        docs = select_sample(docs, args.sample)
        print(f"\nSelected {len(docs)} for sample test:")
        sample_tiers = {}
        sample_pages = 0
        for d in docs:
            tier = d.get("fidelity_risk", "unknown")
            sample_tiers[tier] = sample_tiers.get(tier, 0) + 1
            sample_pages += d.get("page_count", 0) or 0
        for tier in ["critical", "high", "medium"]:
            if tier in sample_tiers:
                print(f"  {tier}: {sample_tiers[tier]}")
        print(f"  ~{sample_pages} pages, est. cost: ${sample_pages * 0.015:.2f}")
    else:
        print(f"\nTotal: ~{total_pages} pages, est. cost: ${total_pages * 0.015:.2f}")

    # --- Dry run ---
    if args.dry_run:
        print(f"\nDocuments to process:")
        for d in docs[:30]:
            risk = d.get("fidelity_risk", "?")
            fid = d.get("fidelity_score", 0) or 0
            qual = d.get("extraction_quality", 0) or 0
            pages = d.get("page_count", "?")
            print(f"  id={d['id']:6d}  {(d.get('letter_id') or ''):20s}  "
                  f"year={d['year_tag']}  {risk:8s}  "
                  f"fidelity={fid:.3f}  quality={qual:.3f}  pages={pages}")
        if len(docs) > 30:
            print(f"  ... and {len(docs) - 30} more")
        return

    # --- Process ---
    client = anthropic.Anthropic(api_key=api_key)
    update_db = not args.no_db_update

    results = []
    total_cost = 0.0
    improved = 0
    degraded = 0
    errors = 0

    start_time = time.time()

    for i, doc in enumerate(docs, 1):
        letter_id = doc.get("letter_id") or f"doc#{doc['id']}"
        tier = doc.get("fidelity_risk", "?")
        print(f"[{i}/{len(docs)}] {letter_id} ({doc['year_tag']}, {tier})", end="", flush=True)

        try:
            result = process_document(client, doc, update_db=update_db)
        except Exception as e:
            result = {"doc_id": doc["id"], "letter_id": letter_id, "error": str(e)}
        results.append(result)

        if result.get("error"):
            print(f" — ERROR: {result['error']}")
            errors += 1
        else:
            delta = result.get("quality_delta")
            old_q = result.get("old_quality", 0) or 0
            new_q = result["new_quality"]
            sim = result.get("old_vs_new_similarity")
            cost = result["cost"]
            total_cost += cost

            delta_str = f"{delta:+.3f}" if delta is not None else "n/a"
            sim_str = f"{sim:.3f}" if sim is not None else "n/a"

            status = "OK"
            if delta is not None:
                if delta > 0:
                    improved += 1
                    status = "IMPROVED"
                elif delta < -0.05:
                    degraded += 1
                    status = "DEGRADED"

            saved_str = " [saved]" if result.get("saved") else " [dry]"
            print(f" — quality: {old_q:.3f}→{new_q:.3f} ({delta_str}) "
                  f"sim={sim_str} ${cost:.4f} {status}{saved_str}")

        # Small delay between documents
        time.sleep(REQUEST_DELAY)

    elapsed = time.time() - start_time

    # --- Report ---
    report = {
        "run_time": datetime.now(timezone.utc).isoformat(),
        "mode": "sample" if args.sample else "all",
        "tier_filter": args.tier,
        "total_processed": len(docs),
        "improved": improved,
        "degraded": degraded,
        "errors": errors,
        "total_cost": round(total_cost, 4),
        "elapsed_seconds": round(elapsed, 1),
        "db_updated": update_db,
        "results": results,
    }

    report_path = SAMPLE_REPORT_PATH if args.sample else REPORT_PATH
    report_path.parent.mkdir(parents=True, exist_ok=True)
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2)

    print(f"\n{'=' * 65}")
    print(f"HAIKU RE-OCR {'SAMPLE' if args.sample else 'BATCH'} SUMMARY")
    print(f"{'=' * 65}")
    print(f"  Processed:    {len(docs)}")
    print(f"  Improved:     {improved}")
    print(f"  Degraded:     {degraded}")
    print(f"  Errors:       {errors}")
    print(f"  Total cost:   ${total_cost:.4f}")
    print(f"  Elapsed:      {elapsed:.0f}s ({elapsed/60:.1f}m)")
    print(f"  Report:       {report_path}")

    if results:
        quality_deltas = [r["quality_delta"] for r in results if r.get("quality_delta") is not None]
        if quality_deltas:
            avg_delta = sum(quality_deltas) / len(quality_deltas)
            print(f"  Avg quality Δ: {avg_delta:+.4f}")

        new_qualities = [r["new_quality"] for r in results if r.get("new_quality") is not None]
        if new_qualities:
            avg_new = sum(new_qualities) / len(new_qualities)
            print(f"  Avg new quality: {avg_new:.4f}")


if __name__ == "__main__":
    main()
