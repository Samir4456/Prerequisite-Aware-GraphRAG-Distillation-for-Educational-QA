"""
src/teacher/build_instruction_set.py

Builds Alpaca-format instruction datasets for Qwen2.5 fine-tuning.

Supports all variants from the instruction set strategy doc:
  - gold             : MetaQA gold answers (no API cost, fast)
  - teacher_answer   : teacher answer only
  - teacher_evidence : teacher evidence + answer
  - hybrid           : teacher evidence + MetaQA gold answer (recommended)

Retrieval modes:
  - rag      : FAISS retrieved chunks only
  - graphrag : graph triples + FAISS chunks (recommended)
  - graph    : graph triples only

Teacher providers:
  - openai   : GPT-4o, GPT-4o-mini (needs OPENAI_API_KEY)
  - deepseek : DeepSeek-V3, DeepSeek-R1 (needs DEEPSEEK_API_KEY)

Usage examples:

  # Gold GraphRAG (free, no API)
  python src/teacher/build_instruction_set.py \
      --mode graphrag --label_source gold \
      --samples_per_hop 500 \
      --output_path data/processed/instruction_pairs/train_graphrag_gold.json

  # Hybrid with GPT-4o-mini
  python src/teacher/build_instruction_set.py \
      --mode graphrag --label_source hybrid \
      --samples_per_hop 500 \
      --teacher_provider openai \
      --teacher_model gpt-4o-mini \
      --output_path data/processed/instruction_pairs/train_graphrag_hybrid_openai.json

  # Hybrid with DeepSeek-V3 (much cheaper)
  python src/teacher/build_instruction_set.py \
      --mode graphrag --label_source hybrid \
      --samples_per_hop 500 \
      --teacher_provider deepseek \
      --teacher_model deepseek-chat \
      --output_path data/processed/instruction_pairs/train_graphrag_hybrid_deepseek.json
"""

import argparse
import json
import os
import random
import sys
import time
from pathlib import Path

from tqdm import tqdm

# ── path setup ────────────────────────────────────────────────────────────────
ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(ROOT / "src" / "data"))
sys.path.insert(0, str(ROOT / "src" / "graph"))
sys.path.insert(0, str(ROOT / "src" / "retrieval"))

from load_kb import load_kb
from load_metaqa import load_all_splits
from entity_extract import extract_topic_entity, clean_question
from subgraph import get_subgraph
from serialize import serialize_triples
from faiss_index import load_index
from embedder import embed_single
import numpy as np


# ─────────────────────────────────────────────
# Provider config
# ─────────────────────────────────────────────

PROVIDER_CONFIG = {
    "openai": {
        "base_url": None,                          # uses default OpenAI endpoint
        "api_key_env": "OPENAI_API_KEY",
        "default_model": "gpt-4o-mini",
        "models": ["gpt-4o", "gpt-4o-mini", "gpt-4.1-mini"],
    },
    "deepseek": {
        "base_url": "https://api.deepseek.com",
        "api_key_env": "DEEPSEEK_API_KEY",
        "default_model": "deepseek-chat",
        "models": ["deepseek-chat", "deepseek-reasoner"],
        # deepseek-chat     = DeepSeek-V3  (~$0.27/1M input, $1.10/1M output)
        # deepseek-reasoner = DeepSeek-R1  (more expensive, chain-of-thought)
    },
}


# ─────────────────────────────────────────────
# Retrieval helpers
# ─────────────────────────────────────────────

def retrieve_chunks(question: str, index, corpus: list, k: int = 5) -> list[str]:
    q_vec = embed_single(question).astype(np.float32)
    scores, indices = index.search(q_vec, k)
    return [corpus[i] for i in indices[0] if i < len(corpus)]


def build_context(
    item: dict,
    adjacency: dict,
    index,
    corpus: list,
    mode: str,
    hops: int,
    k: int,
    max_triples: int,
) -> tuple[str, str]:
    """Build graph text and retrieved chunks for one QA pair."""
    question = item['question']
    entity = item.get('topic_entity') or extract_topic_entity(question) or ''

    graph_text = ""
    if mode in ("graph", "graphrag") and entity:
        subgraph = get_subgraph(entity, adjacency, hops=hops, max_triples=max_triples)
        graph_text = serialize_triples(subgraph, style="arrow")

    chunks_text = ""
    if mode in ("rag", "graphrag"):
        chunks = retrieve_chunks(question, index, corpus, k=k)
        chunks_text = "\n".join(f"- {c}" for c in chunks)

    return graph_text, chunks_text


# ─────────────────────────────────────────────
# Alpaca builder
# ─────────────────────────────────────────────

INSTRUCTION_ANSWER_ONLY = (
    "Answer the question using the retrieved context and knowledge graph. "
    "Return only the answer entity or entities separated by |."
)

INSTRUCTION_EVIDENCE = (
    "Answer the question using the retrieved context and knowledge graph. "
    "First list the supporting evidence from the graph or context, "
    "then give the final answer."
)


def build_input_text(question: str, graph_text: str, chunks_text: str, mode: str) -> str:
    parts = []
    if mode in ("graph", "graphrag") and graph_text:
        parts.append(f"Knowledge Graph:\n{graph_text}")
    if mode in ("rag", "graphrag") and chunks_text:
        parts.append(f"Retrieved Context:\n{chunks_text}")
    parts.append(f"Question: {clean_question(question)}")
    return "\n\n".join(parts)


def build_gold_output(answers: list[str]) -> str:
    return " | ".join(answers)


def build_evidence_prompt(question: str, graph_text: str, chunks_text: str, mode: str) -> str:
    input_text = build_input_text(question, graph_text, chunks_text, mode)
    return (
        "You are creating training data for a graph-augmented QA model.\n\n"
        f"{input_text}\n\n"
        "Respond in this exact format:\n"
        "Supporting evidence:\n"
        "- <triple or context sentence that supports the answer>\n"
        "- <triple or context sentence that supports the answer>\n\n"
        "Final answer: <answer entity or entities separated by |>"
    )


# ─────────────────────────────────────────────
# Teacher call — supports OpenAI + DeepSeek
# ─────────────────────────────────────────────

def call_teacher(
    prompt: str,
    model: str = "gpt-4o-mini",
    provider: str = "openai",
    max_retries: int = 3,
) -> str:
    """
    Call teacher API and return the response text.
    Supports OpenAI and DeepSeek (OpenAI-compatible).
    """
    try:
        import openai
    except ImportError:
        raise ImportError("pip install openai")

    cfg = PROVIDER_CONFIG.get(provider)
    if cfg is None:
        raise ValueError(f"Unknown provider '{provider}'. Choose from: {list(PROVIDER_CONFIG)}")

    api_key = os.environ.get(cfg["api_key_env"])
    if not api_key:
        raise ValueError(
            f"API key not set. Export {cfg['api_key_env']} environment variable.\n"
            f"  Windows: $env:{cfg['api_key_env']}='your-key-here'\n"
            f"  Linux:   export {cfg['api_key_env']}=your-key-here"
        )

    # Build client — DeepSeek uses same openai SDK with custom base_url
    client_kwargs = {"api_key": api_key}
    if cfg["base_url"]:
        client_kwargs["base_url"] = cfg["base_url"]

    client = openai.OpenAI(**client_kwargs)

    for attempt in range(max_retries):
        try:
            response = client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=300,
                temperature=0.0,
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            if attempt < max_retries - 1:
                wait = 2 ** attempt
                print(f"  [RETRY {attempt+1}] {e} — waiting {wait}s")
                time.sleep(wait)
            else:
                print(f"  [WARN] Teacher call failed after {max_retries} attempts: {e}")
                return ""
    return ""


def parse_teacher_evidence_output(raw: str, gold_answers: list[str], label_source: str) -> str:
    if not raw:
        return build_gold_output(gold_answers)

    if label_source == "hybrid":
        lines = raw.split('\n')
        evidence_lines = []
        for line in lines:
            if line.lower().startswith("final answer:"):
                break
            evidence_lines.append(line)
        evidence_text = "\n".join(evidence_lines).strip()
        gold_str = build_gold_output(gold_answers)
        return f"{evidence_text}\n\nFinal answer: {gold_str}"

    elif label_source == "teacher_evidence":
        return raw

    return raw


# ─────────────────────────────────────────────
# Main dataset builder
# ─────────────────────────────────────────────

def build_dataset(args) -> list[dict]:
    print(f"\nConfiguration:")
    print(f"  mode            : {args.mode}")
    print(f"  label_source    : {args.label_source}")
    print(f"  samples/hop     : {args.samples_per_hop}")
    print(f"  hops            : {args.hops}")
    print(f"  k (FAISS)       : {args.k}")
    print(f"  teacher_provider: {args.teacher_provider}")
    print(f"  teacher_model   : {args.teacher_model}")
    print(f"  seed            : {args.seed}")

    print("\nLoading knowledge base...")
    triples, adjacency = load_kb(args.kb_path)
    print(f"  {len(triples):,} triples  |  {len(adjacency):,} nodes")

    print("Loading FAISS index...")
    index, corpus = load_index(args.index_path, args.corpus_path)

    # Collect samples from all 3 hops
    all_pairs = []
    for hop_num in [1, 2, 3]:
        hop_dir = Path(args.data_dir) / f"{hop_num}hop"
        if not hop_dir.exists():
            print(f"  [WARN] {hop_dir} not found, skipping.")
            continue

        splits = load_all_splits(str(hop_dir))
        pairs = splits[args.split]

        random.seed(args.seed)
        sampled = random.sample(pairs, min(args.samples_per_hop, len(pairs)))

        for p in sampled:
            p['hop_num'] = hop_num
        all_pairs.extend(sampled)
        print(f"  {hop_num}-hop {args.split}: sampled {len(sampled):,} / {len(pairs):,}")

    print(f"\nTotal examples: {len(all_pairs):,}")

    needs_teacher = args.label_source in ("teacher_answer", "teacher_evidence", "hybrid")

    if needs_teacher:
        cfg = PROVIDER_CONFIG.get(args.teacher_provider, {})
        api_key_env = cfg.get("api_key_env", "OPENAI_API_KEY")
        if not os.environ.get(api_key_env):
            raise ValueError(
                f"{api_key_env} not set.\n"
                f"Use --label_source gold for free generation, "
                f"or set the {api_key_env} environment variable."
            )
        print(f"\nUsing teacher: {args.teacher_provider} / {args.teacher_model}")

    instruction_str = (
        INSTRUCTION_EVIDENCE
        if args.label_source in ("teacher_evidence", "hybrid")
        else INSTRUCTION_ANSWER_ONLY
    )

    out_path = Path(args.output_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    instruction_pairs = []
    teacher_calls = 0
    teacher_errors = 0

    for item in tqdm(all_pairs, desc="Building instruction pairs"):
        question = item['question']
        answers = item['answers']
        hop_num = item['hop_num']

        context_hops = max(hop_num, 2)
        graph_text, chunks_text = build_context(
            item, adjacency, index, corpus,
            mode=args.mode,
            hops=context_hops,
            k=args.k,
            max_triples=args.max_triples,
        )

        input_text = build_input_text(question, graph_text, chunks_text, args.mode)

        if args.label_source == "gold":
            output_text = build_gold_output(answers)

        elif args.label_source == "teacher_answer":
            raw = call_teacher(
                f"Answer the question. Return only the answer entities separated by |.\n\n{input_text}",
                model=args.teacher_model,
                provider=args.teacher_provider,
            )
            output_text = raw if raw else build_gold_output(answers)
            teacher_calls += 1
            if not raw:
                teacher_errors += 1
            time.sleep(args.rate_limit_delay)

        elif args.label_source in ("teacher_evidence", "hybrid"):
            prompt = build_evidence_prompt(question, graph_text, chunks_text, args.mode)
            raw = call_teacher(
                prompt,
                model=args.teacher_model,
                provider=args.teacher_provider,
            )
            output_text = parse_teacher_evidence_output(raw, answers, args.label_source)
            teacher_calls += 1
            if not raw:
                teacher_errors += 1
            time.sleep(args.rate_limit_delay)

        else:
            raise ValueError(f"Unknown label_source: {args.label_source}")

        instruction_pairs.append({
            "instruction": instruction_str,
            "input": input_text,
            "output": output_text,
            "metadata": {
                "question": question,
                "gold_answers": answers,
                "hop": hop_num,
                "mode": args.mode,
                "label_source": args.label_source,
                "teacher_provider": args.teacher_provider,
                "teacher_model": args.teacher_model,
            }
        })

        if len(instruction_pairs) % 50 == 0:
            _save(instruction_pairs, out_path)

    _save(instruction_pairs, out_path)

    if needs_teacher:
        print(f"\nTeacher calls  : {teacher_calls}")
        print(f"Teacher errors : {teacher_errors}")
        print(f"Success rate   : {(teacher_calls - teacher_errors) / teacher_calls * 100:.1f}%")

    return instruction_pairs


def _save(pairs: list, path: Path):
    clean = [
        {"instruction": p["instruction"], "input": p["input"], "output": p["output"]}
        for p in pairs
    ]
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(clean, f, indent=2, ensure_ascii=False)

    meta_path = path.parent / (path.stem + "_with_meta.json")
    with open(meta_path, 'w', encoding='utf-8') as f:
        json.dump(pairs, f, indent=2, ensure_ascii=False)

    print(f"  Saved {len(pairs):,} pairs → {path}")


# ─────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Build Alpaca instruction dataset")

    parser.add_argument("--data_dir",    default="data/raw")
    parser.add_argument("--kb_path",     default="data/raw/kb.txt")
    parser.add_argument("--index_path",  default="data/faiss/index.bin")
    parser.add_argument("--corpus_path", default="data/faiss/corpus.pkl")
    parser.add_argument("--split",       default="train",
                        choices=["train", "dev", "test"])
    parser.add_argument("--output_path",
                        default="data/processed/instruction_pairs/train_graphrag_gold.json")
    parser.add_argument("--mode", default="graphrag",
                        choices=["rag", "graph", "graphrag"])
    parser.add_argument("--label_source", default="gold",
                        choices=["gold", "teacher_answer", "teacher_evidence", "hybrid"])
    parser.add_argument("--samples_per_hop", type=int, default=500)
    parser.add_argument("--hops",            type=int, default=2)
    parser.add_argument("--k",               type=int, default=5)
    parser.add_argument("--max_triples",     type=int, default=50)
    parser.add_argument("--seed",            type=int, default=42)

    # Teacher — now supports both providers
    parser.add_argument("--teacher_provider", default="openai",
                        choices=["openai", "deepseek"],
                        help="API provider for teacher generation")
    parser.add_argument("--teacher_model", default="gpt-4o-mini",
                        help="Model name (gpt-4o-mini / deepseek-chat / deepseek-reasoner)")
    parser.add_argument("--rate_limit_delay", type=float, default=0.3)

    args = parser.parse_args()

    pairs = build_dataset(args)

    print(f"\n{'='*50}")
    print(f"Done. {len(pairs):,} instruction pairs saved.")
    print(f"Output: {args.output_path}")
