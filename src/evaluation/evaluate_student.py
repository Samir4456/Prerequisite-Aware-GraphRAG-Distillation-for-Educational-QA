"""
src/evaluation/evaluate_student.py

Evaluate a fine-tuned Qwen2.5 student model on MetaQA test set.
Reports EM and F1 per hop, overall, and latency.
Logs everything to WandB.

Usage:
    # Evaluate GraphRAG gold model
    python src/evaluation/evaluate_student.py \
        --model_path checkpoints/qwen2.5-1.5b-graphrag-gold \
        --data_dir data/raw \
        --kb_path data/raw/kb.txt \
        --mode graphrag \
        --run_name qwen2.5-1.5b-graphrag-gold

    # Evaluate RAG-only model
    python src/evaluation/evaluate_student.py \
        --model_path checkpoints/qwen2.5-1.5b-rag-gold \
        --mode rag \
        --run_name qwen2.5-1.5b-rag-gold
"""

import argparse
import json
import re
import sys
import time
from pathlib import Path

import torch
import wandb
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

sys.path.insert(0, str(Path(__file__).parent.parent / "data"))
sys.path.insert(0, str(Path(__file__).parent.parent / "graph"))
sys.path.insert(0, str(Path(__file__).parent.parent / "retrieval"))

from load_kb import load_kb
from load_metaqa import load_all_splits
from entity_extract import extract_topic_entity, clean_question
from subgraph import get_subgraph
from serialize import serialize_triples
from faiss_index import load_index
from embedder import embed_single
import numpy as np


# ─────────────────────────────────────────────
# Context builder
# ─────────────────────────────────────────────

def build_input(item, adjacency, index, corpus, mode, hops, k=5, max_triples=50):
    question = item['question']
    entity = item.get('topic_entity') or extract_topic_entity(question) or ''

    graph_text = ""
    if mode in ("graph", "graphrag") and entity:
        subgraph = get_subgraph(entity, adjacency, hops=hops, max_triples=max_triples)
        graph_text = serialize_triples(subgraph, style="arrow")

    chunks_text = ""
    if mode in ("rag", "graphrag"):
        q_vec = embed_single(question).astype(np.float32)
        scores, indices = index.search(q_vec, k)
        chunks = [corpus[i] for i in indices[0] if i < len(corpus)]
        chunks_text = "\n".join(f"- {c}" for c in chunks)

    parts = []
    if mode in ("graph", "graphrag") and graph_text:
        parts.append(f"Knowledge Graph:\n{graph_text}")
    if mode in ("rag", "graphrag") and chunks_text:
        parts.append(f"Retrieved Context:\n{chunks_text}")
    parts.append(f"Question: {clean_question(question)}")

    return "\n\n".join(parts)


# ─────────────────────────────────────────────
# Inference
# ─────────────────────────────────────────────

INSTRUCTION = (
    "Answer the question using the retrieved context and knowledge graph. "
    "Return only the answer entity or entities separated by |."
)


def build_prompt(input_text: str, tokenizer) -> str:
    """Format as Qwen2.5 chat prompt."""
    messages = [
        {"role": "system", "content": INSTRUCTION},
        {"role": "user", "content": input_text},
    ]
    return tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )


def parse_answer(raw: str) -> list[str]:
    """
    Parse model output into a list of answer entities.
    Handles: "Answer1 | Answer2", "Final answer: X", plain text.
    """
    # Strip "Final answer:" prefix if present
    raw = re.sub(r'(?i)final answer:\s*', '', raw).strip()
    # Split on pipe
    parts = [p.strip() for p in raw.split('|')]
    # Remove empty strings
    return [p for p in parts if p]


@torch.no_grad()
def run_inference(
    model, tokenizer, prompt: str, device, max_new_tokens: int = 64
) -> tuple[str, float]:
    """Run model inference. Returns (raw_output, latency_ms)."""
    inputs = tokenizer(prompt, return_tensors="pt").to(device)

    t0 = time.time()
    outputs = model.generate(
        **inputs,
        max_new_tokens=max_new_tokens,
        do_sample=False,
        pad_token_id=tokenizer.eos_token_id,
    )
    latency = (time.time() - t0) * 1000

    # Decode only the new tokens
    new_tokens = outputs[0][inputs['input_ids'].shape[1]:]
    raw = tokenizer.decode(new_tokens, skip_special_tokens=True).strip()

    return raw, latency


# ─────────────────────────────────────────────
# Metrics
# ─────────────────────────────────────────────

def exact_match(pred_list: list[str], gold_list: list[str]) -> float:
    pred_set = {p.strip().lower() for p in pred_list}
    gold_set = {g.strip().lower() for g in gold_list}
    return float(pred_set == gold_set)


def f1_score(pred_list: list[str], gold_list: list[str]) -> float:
    pred_set = {p.strip().lower() for p in pred_list}
    gold_set = {g.strip().lower() for g in gold_list}
    if not pred_set or not gold_set:
        return 0.0
    common = pred_set & gold_set
    if not common:
        return 0.0
    precision = len(common) / len(pred_set)
    recall = len(common) / len(gold_set)
    return 2 * precision * recall / (precision + recall)


# ─────────────────────────────────────────────
# Evaluate one hop
# ─────────────────────────────────────────────

def evaluate_hop(
    model, tokenizer, device,
    qa_pairs, adjacency, index, corpus,
    hop_num, mode, n_samples, k, max_triples,
):
    context_hops = max(hop_num, 2)
    pairs = qa_pairs[:n_samples]

    em_scores, f1_scores, latencies = [], [], []
    examples = []

    for item in tqdm(pairs, desc=f"  {hop_num}-hop", leave=False):
        input_text = build_input(
            item, adjacency, index, corpus,
            mode=mode, hops=context_hops, k=k, max_triples=max_triples
        )
        prompt = build_prompt(input_text, tokenizer)
        raw, latency = run_inference(model, tokenizer, prompt, device)
        pred_list = parse_answer(raw)

        em = exact_match(pred_list, item['answers'])
        f1 = f1_score(pred_list, item['answers'])

        em_scores.append(em)
        f1_scores.append(f1)
        latencies.append(latency)

        examples.append({
            "question": item['question'],
            "gold": item['answers'],
            "pred": pred_list,
            "raw_output": raw,
            "em": em,
            "f1": f1,
            "latency_ms": round(latency, 2),
        })

    n = len(em_scores)
    return {
        "EM": round(sum(em_scores) / n, 4) if n else 0,
        "F1": round(sum(f1_scores) / n, 4) if n else 0,
        "latency_ms": round(sum(latencies) / n, 2) if n else 0,
        "n": n,
        "examples": examples,
    }


# ─────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────

def main(args):
    wandb.init(
        project="pocket-graphrag",
        name=args.run_name,
        config=vars(args),
        tags=["student", "evaluation", args.mode],
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\nDevice : {device}")
    if device.type == "cuda":
        print(f"GPU    : {torch.cuda.get_device_name(0)}")

    # Load model
    print(f"\nLoading model from {args.model_path}...")
    tokenizer = AutoTokenizer.from_pretrained(
        args.model_path, trust_remote_code=True
    )
    model = AutoModelForCausalLM.from_pretrained(
        args.model_path,
        torch_dtype=torch.float16,
        device_map="auto",
        trust_remote_code=True,
    )
    model.eval()
    print("  Model loaded.")

    # Load KB and FAISS
    print("\nLoading knowledge base...")
    triples, adjacency = load_kb(args.kb_path)

    print("Loading FAISS index...")
    index, corpus = load_index(args.index_path, args.corpus_path)

    # Evaluate per hop
    all_metrics = {}
    all_examples = []

    print(f"\nEvaluating mode={args.mode} on {args.n_samples} samples per hop...\n")

    for hop_num in [1, 2, 3]:
        hop_dir = Path(args.data_dir) / f"{hop_num}hop"
        if not hop_dir.exists():
            continue

        splits = load_all_splits(str(hop_dir), max_samples=args.n_samples)
        test_pairs = splits['test']

        print(f"{hop_num}-hop test ({len(test_pairs)} samples):")
        results = evaluate_hop(
            model, tokenizer, device,
            test_pairs, adjacency, index, corpus,
            hop_num=hop_num,
            mode=args.mode,
            n_samples=args.n_samples,
            k=args.k,
            max_triples=args.max_triples,
        )

        tag = f"test_{hop_num}hop"
        print(f"  EM={results['EM']}  F1={results['F1']}  "
              f"latency={results['latency_ms']}ms")

        all_metrics[f"{tag}/EM"] = results['EM']
        all_metrics[f"{tag}/F1"] = results['F1']
        all_metrics[f"{tag}/latency_ms"] = results['latency_ms']
        all_examples.extend(results['examples'])

        wandb.log({
            f"{tag}/EM": results['EM'],
            f"{tag}/F1": results['F1'],
            f"{tag}/latency_ms": results['latency_ms'],
        })

    # Overall metrics
    em_vals = [v for k, v in all_metrics.items() if k.endswith("/EM")]
    f1_vals = [v for k, v in all_metrics.items() if k.endswith("/F1")]
    all_metrics["overall/EM"] = round(sum(em_vals) / len(em_vals), 4)
    all_metrics["overall/F1"] = round(sum(f1_vals) / len(f1_vals), 4)
    wandb.log(all_metrics)

    # Save results
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    summary = {
        "model": args.model_path,
        "mode": args.mode,
        "n_samples": args.n_samples,
        **all_metrics,
    }
    with open(out_dir / "eval_results.json", 'w') as f:
        json.dump(summary, f, indent=2)

    with open(out_dir / "eval_examples.json", 'w') as f:
        json.dump(all_examples[:100], f, indent=2, ensure_ascii=False)

    print(f"\n{'='*50}")
    print("STUDENT MODEL RESULTS")
    print(f"{'='*50}")
    for k, v in summary.items():
        if k not in ("model", "mode", "n_samples"):
            print(f"  {k}: {v}")

    print(f"\nResults saved → {out_dir}")
    wandb.finish()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path",  required=True,
                        help="Path to fine-tuned model checkpoint")
    parser.add_argument("--data_dir",    default="data/raw")
    parser.add_argument("--kb_path",     default="data/raw/kb.txt")
    parser.add_argument("--index_path",  default="data/faiss/index.bin")
    parser.add_argument("--corpus_path", default="data/faiss/corpus.pkl")
    parser.add_argument("--mode",        default="graphrag",
                        choices=["rag", "graph", "graphrag"])
    parser.add_argument("--n_samples",   type=int, default=500,
                        help="Test samples per hop")
    parser.add_argument("--k",           type=int, default=5)
    parser.add_argument("--max_triples", type=int, default=50)
    parser.add_argument("--output_dir",  default="results/qwen2.5-eval")
    parser.add_argument("--run_name",    default="qwen2.5-student-eval")
    args = parser.parse_args()
    main(args)
