"""
CLRS Textbook Parsing & Chunking Pipeline
==========================================
Usage:
    1. Place your CLRS PDF at:  data/raw/clrs.pdf
    2. Run:  python clrs_pipeline.py
    3. Output saved to:  data/processed/clrs_chunks.json

Requirements:
    pip install pymupdf langchain tiktoken
"""

import re
import json
import random
from pathlib import Path

import fitz  # PyMuPDF
import tiktoken
from langchain_text_splitters import RecursiveCharacterTextSplitter


# ─── Config ──────────────────────────────────────────────────────────────────

PDF_PATH        = "data/raw/clrs.pdf"
OUTPUT_DIR      = Path("data/processed")
OUTPUT_PATH     = OUTPUT_DIR / "clrs_chunks.json"

CHUNK_SIZE      = 450    # target tokens per chunk
CHUNK_OVERLAP   = 50     # overlap tokens between consecutive chunks
MIN_CHUNK_TOKENS = 60    # discard fragments shorter than this
SOURCE_NAME     = "clrs"

# Pages to skip (0-indexed): front matter, index, bibliography
# Adjust these after inspecting your specific PDF edition
SKIP_PAGES_BEFORE = 20   # skip title page, TOC, preface (~20 pages)
SKIP_PAGES_AFTER  = 30   # skip bibliography + index at the end


# ─── Tokenizer ───────────────────────────────────────────────────────────────

enc = tiktoken.encoding_for_model("gpt-4o")

def token_len(text: str) -> int:
    return len(enc.encode(text))


# ─── Step 1: Extract raw text from PDF ───────────────────────────────────────

def extract_pages(pdf_path: str) -> list[dict]:
    """Extract text from each page using PyMuPDF."""
    doc = fitz.open(pdf_path)
    total = len(doc)
    print(f"  PDF loaded: {total} pages total")

    pages = []
    start = SKIP_PAGES_BEFORE
    end   = total - SKIP_PAGES_AFTER

    for page_num in range(start, end):
        page = doc[page_num]
        text = page.get_text("text")  # plain text mode
        pages.append({
            "page_index": page_num,          # 0-based
            "page_number": page_num + 1,     # 1-based (for display)
            "text": text
        })

    print(f"  Extracted {len(pages)} content pages (skipped {start} front + {SKIP_PAGES_AFTER} back)")
    return pages


# ─── Step 2: Clean raw text ───────────────────────────────────────────────────

# Lines that match any of these patterns will be dropped
NOISE_PATTERNS = [
    r'^\s*\d+\s*$',                                   # standalone page numbers
    r'^\s*CHAPTER\s+\d+\s*$',                         # chapter header lines
    r'^\s*Introduction to Algorithms\s*$',            # running header
    r'^\s*(CLRS|Cormen|Leiserson|Rivest|Stein)\s*$',  # author headers
    r'^\s*\d+\.\d+\s+[A-Z][A-Z\s]+$',                # section number + ALL CAPS heading
    r'^\s*Figure \d+[\.\d]*.*$',                      # figure captions
    r'^\s*Exercises\s*$',                             # exercise section headers
    r'^\s*Problems\s*$',                              # problem section headers
]
NOISE_RE = re.compile('|'.join(NOISE_PATTERNS), re.IGNORECASE | re.MULTILINE)


def clean_text(text: str) -> str:
    """Remove noise lines and normalise whitespace."""
    # Drop noise lines
    lines = text.split('\n')
    clean_lines = []
    for line in lines:
        if not NOISE_RE.match(line):
            clean_lines.append(line)
    text = '\n'.join(clean_lines)

    # Collapse 3+ blank lines into 2
    text = re.sub(r'\n{3,}', '\n\n', text)
    # Normalise spaces within lines
    text = re.sub(r'[ \t]+', ' ', text)
    # Remove lines that are only punctuation/symbols (PDF artifact lines)
    text = re.sub(r'^\s*[^\w\s]{3,}\s*$', '', text, flags=re.MULTILINE)

    return text.strip()


def detect_chapter(text: str, page_number: int) -> str:
    """Try to extract a chapter label from the first 300 chars of the page text."""
    snippet = text[:300]
    match = re.search(
        r'(Chapter\s+\d+[\.\d]*\s*[:\-–]?\s*.{0,60})',
        snippet,
        re.IGNORECASE
    )
    if match:
        return match.group(1).strip()[:80]  # cap at 80 chars
    return f"page_{page_number}"


def clean_pages(pages: list[dict]) -> list[dict]:
    cleaned = []
    skipped = 0
    for p in pages:
        text = clean_text(p["text"])
        if token_len(text) < MIN_CHUNK_TOKENS:
            skipped += 1
            continue
        cleaned.append({
            **p,
            "text": text,
            "chapter": detect_chapter(text, p["page_number"])
        })
    print(f"  Cleaned {len(cleaned)} pages, dropped {skipped} near-empty pages")
    return cleaned


# ─── Step 3: Chunk ────────────────────────────────────────────────────────────

def build_splitter() -> RecursiveCharacterTextSplitter:
    return RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        length_function=token_len,
        # Try to split on paragraph breaks first, then sentences, then words
        separators=["\n\n", "\n", ". ", "! ", "? ", "; ", " ", ""]
    )


def chunk_pages(pages: list[dict]) -> list[dict]:
    splitter = build_splitter()
    chunks = []
    chunk_id = 0

    for page in pages:
        page_chunks = splitter.split_text(page["text"])
        for chunk_text in page_chunks:
            tokens = token_len(chunk_text)
            if tokens < MIN_CHUNK_TOKENS:
                continue
            chunks.append({
                "chunk_id":    f"{SOURCE_NAME}_chunk_{chunk_id:05d}",
                "source":      SOURCE_NAME,
                "page_number": page["page_number"],
                "chapter":     page["chapter"],
                "text":        chunk_text.strip(),
                "token_count": tokens,
                "char_count":  len(chunk_text),
            })
            chunk_id += 1

    print(f"  Created {len(chunks)} chunks")
    return chunks


# ─── Step 4: Validate ─────────────────────────────────────────────────────────

def validate_chunks(chunks: list[dict]) -> None:
    if not chunks:
        print("  WARNING: No chunks produced — check PDF path and skip page settings")
        return

    token_counts = [c["token_count"] for c in chunks]
    avg   = sum(token_counts) / len(token_counts)
    min_t = min(token_counts)
    max_t = max(token_counts)

    under_100 = sum(1 for t in token_counts if t < 100)
    over_500  = sum(1 for t in token_counts if t > 500)

    print(f"\n  Chunk statistics:")
    print(f"    Total chunks : {len(chunks)}")
    print(f"    Avg tokens   : {avg:.0f}")
    print(f"    Min tokens   : {min_t}")
    print(f"    Max tokens   : {max_t}")
    print(f"    Under 100 tok: {under_100}  (consider raising MIN_CHUNK_TOKENS if high)")
    print(f"    Over 500 tok : {over_500}   (consider lowering CHUNK_SIZE if high)")

    # Overlap sanity: check consecutive chunks share some text
    overlap_ok = 0
    for i in range(min(10, len(chunks) - 1)):
        tail = chunks[i]["text"][-100:]
        head = chunks[i+1]["text"][:100]
        shared = set(tail.split()) & set(head.split())
        if shared:
            overlap_ok += 1
    print(f"    Overlap check: {overlap_ok}/10 consecutive pairs share tokens (expect >5)")

    # Print 3 random sample chunks
    print(f"\n  Sample chunks:")
    for sample in random.sample(chunks, min(3, len(chunks))):
        print(f"\n  --- {sample['chunk_id']} | {sample['chapter']} | {sample['token_count']} tokens ---")
        print(f"  {sample['text'][:300]}{'...' if len(sample['text']) > 300 else ''}")


# ─── Step 5: Save ────────────────────────────────────────────────────────────

def save_chunks(chunks: list[dict]) -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        json.dump(chunks, f, indent=2, ensure_ascii=False)
    size_kb = OUTPUT_PATH.stat().st_size / 1024
    print(f"\n  Saved to {OUTPUT_PATH}  ({size_kb:.0f} KB)")


# ─── Main ────────────────────────────────────────────────────────────────────

def main():
    print("=" * 55)
    print("CLRS Parsing & Chunking Pipeline")
    print("=" * 55)

    if not Path(PDF_PATH).exists():
        print(f"\n  ERROR: PDF not found at '{PDF_PATH}'")
        print(f"  Create the directory and place your CLRS PDF there:")
        print(f"    mkdir -p data/raw")
        print(f"    cp /path/to/your/clrs.pdf data/raw/clrs.pdf")
        return

    print(f"\n[1/4] Extracting text from {PDF_PATH} ...")
    pages = extract_pages(PDF_PATH)

    print(f"\n[2/4] Cleaning extracted text ...")
    pages = clean_pages(pages)

    print(f"\n[3/4] Chunking into {CHUNK_SIZE}-token passages ...")
    chunks = chunk_pages(pages)

    print(f"\n[4/4] Validating and saving ...")
    validate_chunks(chunks)
    save_chunks(chunks)

    print("\nDone. Next step: run concept extraction with GPT-4o on these chunks.")
    print("=" * 55)


if __name__ == "__main__":
    main()