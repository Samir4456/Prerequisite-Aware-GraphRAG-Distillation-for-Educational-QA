"""
QA Pair Generation Pipeline — CLRS (Llama 3.1 8B, local)
==========================================================
Runs fully offline using Llama 3.1 8B via Ollama.
No API key needed. GPU recommended (8GB VRAM minimum for 8B Q4).

Setup (one-time):
    # 1. Install Ollama
    curl -fsSL https://ollama.com/install.sh | sh

    # 2. Pull the model (~4.7 GB download for Q4_K_M)
    ollama pull llama3.1:8b

    # 3. Install Python deps
    pip install ollama tiktoken tqdm

Usage:
    # Make sure Ollama is running first
    ollama serve &

    python qa_generation_llama.py

    # Dry-run (prints prompts, no model calls)
    python qa_generation_llama.py --dry-run

    # Resume after crash
    python qa_generation_llama.py --resume

    # Use a different model (e.g. mistral, gemma2)
    python qa_generation_llama.py --model mistral:7b

Output:
    data/processed/clrs_qa_pairs_llama.json

Notes on quality vs GPT-4o:
    - Single-hop questions are nearly as good
    - Multi-hop questions are weaker — Llama 3.1 8B sometimes
      generates questions that only require one passage
    - Validation is less reliable (model validates its own outputs)
    - Recommend: use this for the bulk of your dataset (~80%),
      use GPT-4o selectively for hard multi-hop pairs (~20%)
"""

import argparse
import json
import random
import re
import time
from pathlib import Path

import tiktoken
from tqdm import tqdm

try:
    import ollama
except ImportError:
    print("ERROR: ollama not installed. Run: pip install ollama")
    raise


# ─── Config ──────────────────────────────────────────────────────────────────

CHUNKS_PATH      = Path("data/processed/clrs_chunks.json")
OUTPUT_PATH      = Path("data/processed/clrs_qa_pairs_llama.json")
CHECKPOINT_PATH  = Path("data/processed/clrs_qa_checkpoint_llama.json")

DEFAULT_MODEL    = "llama3.1:8b"

# Generation settings
TEMPERATURE      = 0.7
NUM_CTX          = 4096    # context window — 8B handles 4096 comfortably on 8GB VRAM
MAX_TOKENS_OUT   = 800     # max output tokens per call

# Pipeline settings
MULTIHOP_WINDOW  = 3       # chunk window for multi-hop pairing
SLEEP_BETWEEN    = 0.2     # seconds between calls (local = no rate limit, but helps stability)
MAX_CHUNKS       = 200     # set to None for full run
MIN_CHUNK_TOKENS = 80      # skip very short chunks

# Retries on bad JSON
MAX_JSON_RETRIES = 2


# ─── Tokenizer ───────────────────────────────────────────────────────────────

enc = tiktoken.encoding_for_model("gpt-4o")  # good enough for token counting

def count_tokens(text: str) -> int:
    return len(enc.encode(text))


# ─── PDF ligature cleanup ─────────────────────────────────────────────────────
# PDFs bake ligatures into single Unicode codepoints (ﬁ, ﬂ, ﬀ, ﬃ, ﬄ).
# These survive extraction and break cp1252 on Windows. Normalize them out.
LIGATURE_MAP = {
    "\ufb00": "ff",
    "\ufb01": "fi",
    "\ufb02": "fl",
    "\ufb03": "ffi",
    "\ufb04": "ffl",
    "\ufb05": "st",
    "\ufb06": "st",
}

def clean_ligatures(text: str) -> str:
    for lig, replacement in LIGATURE_MAP.items():
        text = text.replace(lig, replacement)
    return text


# ─── Prompts — tuned for Llama 3.1 instruction format ────────────────────────
# Key differences from GPT-4o prompts:
#   - More explicit JSON schema (Llama needs more hand-holding)
#   - "Return ONLY the JSON object" repeated at end (reduces prose wrapping)
#   - Shorter passages — Llama 3.1 8B degrades with very long contexts

SINGLE_HOP_SYSTEM = "You are a precise JSON-generating assistant. You only output valid JSON. Never add explanation or markdown."

SINGLE_HOP_PROMPT = """Read this passage from an algorithms textbook (CLRS) and generate 2 exam questions.

Rules:
- Each question must be answerable using ONLY the passage below
- No yes/no questions
- Focus on: definitions, properties, time complexity, algorithm steps, or comparisons
- answer_span must be a short exact phrase copied from the passage

Passage:
\"\"\"{chunk_text}\"\"\"

Return ONLY this JSON object, nothing else:
{{
  "questions": [
    {{
      "question": "your question here",
      "answer": "complete answer here",
      "answer_span": "exact short phrase from passage",
      "question_type": "single_hop"
    }},
    {{
      "question": "your second question here",
      "answer": "complete answer here",
      "answer_span": "exact short phrase from passage",
      "question_type": "single_hop"
    }}
  ]
}}"""


MULTI_HOP_SYSTEM = "You are a precise JSON-generating assistant. You only output valid JSON. Never add explanation or markdown."

MULTI_HOP_PROMPT = """Read these two passages from an algorithms textbook (CLRS) and write 1 exam question.

Rule: The question MUST require information from BOTH passages to answer.
Good multi-hop questions ask: how do two algorithms compare, what prerequisite concept
enables another, or how does a property from one passage apply in the other.

Passage A (id: {chunk_id_a}):
\"\"\"{chunk_text_a}\"\"\"

Passage B (id: {chunk_id_b}):
\"\"\"{chunk_text_b}\"\"\"

Return ONLY this JSON object, nothing else:
{{
  "questions": [
    {{
      "question": "your multi-hop question here",
      "answer": "answer that draws on both passages",
      "supporting_chunks": ["{chunk_id_a}", "{chunk_id_b}"],
      "question_type": "multi_hop"
    }}
  ]
}}"""


VALIDATION_SYSTEM = "You are a precise JSON-generating assistant. You only output valid JSON."

VALIDATION_PROMPT = """Does the answer correctly answer the question based on the passage?

Passage:
\"\"\"{chunk_text}\"\"\"

Question: {question}
Answer: {answer}

Return ONLY this JSON, nothing else:
{{
  "is_valid": true,
  "reason": "brief reason"
}}

If the answer is wrong or not supported, set is_valid to false."""


# ─── Ollama call ─────────────────────────────────────────────────────────────

def call_llama(
    prompt: str,
    system: str,
    model: str,
    dry_run: bool = False,
) -> str | None:
    """Single Ollama completion call with error handling."""
    if dry_run:
        print(f"\n[DRY RUN] System: {system[:80]}")
        print(f"[DRY RUN] Prompt (first 400 chars):\n{prompt[:400]}\n")
        return None

    time.sleep(SLEEP_BETWEEN)
    try:
        response = ollama.chat(
            model=model,
            messages=[
                {"role": "system", "content": system},
                {"role": "user",   "content": prompt},
            ],
            options={
                "temperature": TEMPERATURE,
                "num_ctx":     NUM_CTX,
                "num_predict": MAX_TOKENS_OUT,
            },
        )
        return response["message"]["content"]
    except Exception as e:
        print(f"  Ollama error: {e}")
        print("  Is Ollama running? Try: ollama serve")
        return None


# ─── JSON parsing (Llama sometimes wraps output in prose) ────────────────────

def parse_json_response(raw: str) -> dict | None:
    if not raw:
        return None

    raw = raw.strip()

    # Strip markdown fences
    if "```" in raw:
        match = re.search(r'```(?:json)?\s*([\s\S]*?)```', raw)
        if match:
            raw = match.group(1).strip()

    # Try direct parse first
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        pass

    # Llama sometimes adds prose before/after JSON — extract the {...} block
    match = re.search(r'\{[\s\S]*\}', raw)
    if match:
        try:
            return json.loads(match.group(0))
        except json.JSONDecodeError:
            pass

    return None


def call_with_retry(prompt: str, system: str, model: str, dry_run: bool) -> dict | None:
    """Call Llama and retry up to MAX_JSON_RETRIES times on bad JSON."""
    for attempt in range(MAX_JSON_RETRIES + 1):
        raw = call_llama(prompt, system, model, dry_run)
        if raw is None:
            return None
        parsed = parse_json_response(raw)
        if parsed is not None:
            return parsed
        if attempt < MAX_JSON_RETRIES:
            print(f"  Bad JSON (attempt {attempt+1}), retrying...")
    print(f"  Giving up after {MAX_JSON_RETRIES+1} attempts")
    return None


# ─── QA generators ───────────────────────────────────────────────────────────

def generate_single_hop(chunk: dict, model: str, dry_run: bool) -> list[dict]:
    # Truncate very long chunks — Llama 3.1 8B handles ~600 tokens of passage well
    text = clean_ligatures(chunk["text"])
    if count_tokens(text) > 600:
        # Keep first 600 tokens worth of characters (~2400 chars)
        text = text[:2400]

    prompt = SINGLE_HOP_PROMPT.format(chunk_text=text)
    parsed = call_with_retry(prompt, SINGLE_HOP_SYSTEM, model, dry_run)

    if not parsed or "questions" not in parsed:
        return []

    results = []
    for i, q in enumerate(parsed["questions"]):
        if not q.get("question") or not q.get("answer"):
            continue
        # Basic quality filter — skip very short answers
        if len(q["answer"]) < 15:
            continue
        results.append({
            "qa_id":          f"{chunk['chunk_id']}_sh_{i}",
            "question":       q["question"].strip(),
            "answer":         q["answer"].strip(),
            "answer_span":    q.get("answer_span", "").strip(),
            "question_type":  "single_hop",
            "source_chunks":  [chunk["chunk_id"]],
            "source_page":    chunk["page_number"],
            "chapter":        chunk["chapter"],
            "generated_by":   model,
        })
    return results


def generate_multi_hop(chunk_a: dict, chunk_b: dict, model: str, dry_run: bool) -> list[dict]:
    # Truncate both passages
    text_a = clean_ligatures(chunk_a["text"])[:1800]
    text_b = clean_ligatures(chunk_b["text"])[:1800]

    prompt = MULTI_HOP_PROMPT.format(
        chunk_id_a=chunk_a["chunk_id"],
        chunk_text_a=text_a,
        chunk_id_b=chunk_b["chunk_id"],
        chunk_text_b=text_b,
    )
    parsed = call_with_retry(prompt, MULTI_HOP_SYSTEM, model, dry_run)

    if not parsed or "questions" not in parsed:
        return []

    results = []
    for i, q in enumerate(parsed["questions"]):
        if not q.get("question") or not q.get("answer"):
            continue
        results.append({
            "qa_id":          f"{chunk_a['chunk_id']}_mh_{i}",
            "question":       q["question"].strip(),
            "answer":         q["answer"].strip(),
            "question_type":  "multi_hop",
            "source_chunks":  [chunk_a["chunk_id"], chunk_b["chunk_id"]],
            "source_page":    chunk_a["page_number"],
            "chapter":        chunk_a["chapter"],
            "generated_by":   model,
        })
    return results


def validate_qa(chunk: dict, qa: dict, model: str, dry_run: bool) -> bool:
    """Quick self-validation — only for single-hop."""
    if qa["question_type"] != "single_hop":
        return True
    if dry_run:
        return True

    text = clean_ligatures(chunk["text"])[:2000]
    prompt = VALIDATION_PROMPT.format(
        chunk_text=text,
        question=qa["question"],
        answer=qa["answer"],
    )
    parsed = call_with_retry(prompt, VALIDATION_SYSTEM, model, dry_run)
    if not parsed:
        return True  # keep on error

    valid = parsed.get("is_valid", True)
    if not valid:
        print(f"  Rejected: {qa['question'][:70]}...")
    return valid


# ─── Checkpoint ──────────────────────────────────────────────────────────────

def load_checkpoint() -> tuple[list[dict], set[str]]:
    if CHECKPOINT_PATH.exists():
        with open(CHECKPOINT_PATH, encoding="utf-8") as f:
            saved = json.load(f)
        done = {qa["source_chunks"][0] for qa in saved}
        print(f"  Resuming: {len(saved)} pairs done, {len(done)} chunks processed")
        return saved, done
    return [], set()


def save_checkpoint(qa_pairs: list[dict]) -> None:
    CHECKPOINT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(CHECKPOINT_PATH, "w", encoding="utf-8") as f:
        json.dump(qa_pairs, f, indent=2, ensure_ascii=False)


# ─── Ollama health check ──────────────────────────────────────────────────────

def check_ollama(model: str) -> bool:
    """Verify Ollama is running and model is available."""
    try:
        response = ollama.list()
        # Handle both dict-style (old lib) and object-style (new lib) responses
        if isinstance(response, dict):
            raw_models = response.get("models", [])
        else:
            raw_models = getattr(response, "models", [])

        available = []
        for m in raw_models:
            if isinstance(m, dict):
                available.append(m.get("name", m.get("model", "")))
            else:
                name = getattr(m, "model", None) or getattr(m, "name", None) or str(m)
                available.append(name)

        model_base = model.split(":")[0]
        found = any(model_base in m for m in available)
        if not found:
            print(f"\nWARNING: Model '{model}' not found in Ollama.")
            print(f"Available models: {available}")
            print(f"Pull it with: ollama pull {model}")
            return False
        print(f"  Ollama OK — model '{model}' is available")
        return True
    except Exception as e:
        print(f"\nERROR: Cannot connect to Ollama: {e}")
        print("Start Ollama with: ollama serve")
        return False


# ─── Main ────────────────────────────────────────────────────────────────────

def main(dry_run: bool = False, resume: bool = False, model: str = DEFAULT_MODEL):
    print("=" * 55)
    print(f"QA Generation — CLRS  [{model}]")
    print("=" * 55)

    # Load chunks
    if not CHUNKS_PATH.exists():
        print(f"\nERROR: {CHUNKS_PATH} not found. Run clrs_pipeline.py first.")
        return

    with open(CHUNKS_PATH, encoding="utf-8") as f:
        all_chunks = json.load(f)

    chunks = all_chunks[:MAX_CHUNKS] if MAX_CHUNKS else all_chunks
    print(f"\nLoaded {len(all_chunks)} chunks — processing {len(chunks)}")

    # Health check
    if not dry_run and not check_ollama(model):
        return

    # Resume
    qa_pairs, done_ids = load_checkpoint() if resume else ([], set())

    # Filter chunks that still need processing
    todo_chunks = [c for c in chunks if c["chunk_id"] not in done_ids]
    print(f"Chunks to process: {len(todo_chunks)}")

    # ── Single-hop ────────────────────────────────────────────────────────────
    print(f"\n[1/3] Single-hop generation ...")
    sh_count = 0

    for chunk in tqdm(todo_chunks, desc="Single-hop"):
        if count_tokens(chunk["text"]) < MIN_CHUNK_TOKENS:
            continue
        if dry_run and sh_count >= 2:
            break

        new_qa = generate_single_hop(chunk, model, dry_run)

        validated = []
        for qa in new_qa:
            if validate_qa(chunk, qa, model, dry_run):
                validated.append(qa)

        qa_pairs.extend(validated)
        sh_count += len(validated)

        if not dry_run and sh_count % 30 == 0 and sh_count > 0:
            save_checkpoint(qa_pairs)
            print(f"  Checkpoint saved ({sh_count} single-hop so far)")

    print(f"  Single-hop done: {sh_count} pairs")

    # ── Multi-hop ─────────────────────────────────────────────────────────────
    print(f"\n[2/3] Multi-hop generation ...")
    mh_count = 0

    pairs_to_generate = []
    for i in range(0, len(chunks) - MULTIHOP_WINDOW, MULTIHOP_WINDOW):
        window = chunks[i : i + MULTIHOP_WINDOW]
        if len(window) >= 2:
            a, b = random.sample(window, 2)
            pairs_to_generate.append((a, b))

    for chunk_a, chunk_b in tqdm(pairs_to_generate, desc="Multi-hop"):
        if dry_run and mh_count >= 1:
            break

        new_qa = generate_multi_hop(chunk_a, chunk_b, model, dry_run)
        qa_pairs.extend(new_qa)
        mh_count += len(new_qa)

    print(f"  Multi-hop done: {mh_count} pairs")

    # ── Save ─────────────────────────────────────────────────────────────────
    print(f"\n[3/3] Saving ...")
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        json.dump(qa_pairs, f, indent=2, ensure_ascii=False)

    if CHECKPOINT_PATH.exists() and not dry_run:
        CHECKPOINT_PATH.unlink()

    # ── Summary ───────────────────────────────────────────────────────────────
    total = len(qa_pairs)
    sh = sum(1 for q in qa_pairs if q["question_type"] == "single_hop")
    mh = sum(1 for q in qa_pairs if q["question_type"] == "multi_hop")

    print(f"\n  Results:")
    print(f"    Total pairs  : {total}")
    print(f"    Single-hop   : {sh}")
    print(f"    Multi-hop    : {mh}")
    print(f"    Saved to     : {OUTPUT_PATH}")
    print(f"    Cost         : $0.00  (fully local)")

    print(f"\n  Sample QA pairs:")
    for qa in random.sample(qa_pairs, min(3, total)):
        print(f"\n  [{qa['question_type']}] {qa['chapter']}")
        print(f"  Q: {qa['question']}")
        print(f"  A: {qa['answer'][:200]}")

    print("\nDone.")
    print("=" * 55)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true",
                        help="Print prompts without calling the model")
    parser.add_argument("--resume", action="store_true",
                        help="Resume from checkpoint after crash")
    parser.add_argument("--model", default=DEFAULT_MODEL,
                        help=f"Ollama model name (default: {DEFAULT_MODEL})")
    args = parser.parse_args()
    main(dry_run=args.dry_run, resume=args.resume, model=args.model)