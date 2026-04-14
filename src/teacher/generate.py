"""
src/teacher/generate.py

Stage 2 and Stage 3 runner:
- samples fresh random questions from 1-hop, 2-hop, and 3-hop MetaQA splits
- runs RAG and GraphRAG pipelines
- sends the prompt to a modular answer generator
- evaluates EM/F1 and saves detailed outputs
"""

from __future__ import annotations

import argparse
import json
import random
import sys
import time
from pathlib import Path

from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent.parent / "data"))
sys.path.insert(0, str(Path(__file__).parent.parent / "evaluation"))
sys.path.insert(0, str(Path(__file__).parent.parent / "graph"))
sys.path.insert(0, str(Path(__file__).parent.parent / "retrieval"))
sys.path.insert(0, str(Path(__file__).parent))

from answer_generator import build_answer_generator
from metrics import exact_match, f1_score
from load_kb import load_kb
from load_metaqa import load_all_splits, sample_qa_pairs
from faiss_index import load_or_build_index, triples_to_corpus
from retrieve import run_rag_pipeline, recall_at_k
from subgraph import answer_in_subgraph


def evaluate_mode_on_samples(
    qa_pairs: list[dict],
    mode: str,
    hop_depth: int,
    adjacency: dict,
    faiss_index,
    corpus: list[str],
    answer_generator,
    k: int,
    max_triples: int,
) -> tuple[list[dict], dict]:
    """
    Run one answering mode over one sampled hop subset.
    """
    outputs = []
    em_scores = []
    f1_scores = []
    latencies = []
    recall_hits = 0
    graph_hits = 0

    for item in tqdm(qa_pairs, desc=f"{mode} {hop_depth}-hop", leave=False):
        pipeline_result = run_rag_pipeline(
            item["question"],
            adjacency,
            faiss_index,
            corpus,
            hops=hop_depth,
            k=k,
            mode=mode,
            max_triples=max_triples,
        )

        start = time.perf_counter()
        answer_result = answer_generator.generate_answer(pipeline_result["prompt"])
        latency_ms = (time.perf_counter() - start) * 1000

        prediction = answer_result.answer
        em = exact_match(prediction, item["answers"])
        f1 = f1_score(prediction, item["answers"])

        em_scores.append(em)
        f1_scores.append(f1)
        latencies.append(latency_ms)

        if mode in ("rag", "graphrag") and recall_at_k(item["answers"], pipeline_result["retrieved_chunks"]):
            recall_hits += 1
        if mode == "graphrag" and answer_in_subgraph(item["answers"], pipeline_result["subgraph"]):
            graph_hits += 1

        outputs.append(
            {
                "question": item["question"],
                "clean_question": pipeline_result["clean_question"],
                "topic_entity": pipeline_result["topic_entity"],
                "gold_answers": item["answers"],
                "prediction": prediction,
                "em": em,
                "f1": round(f1, 4),
                "latency_ms": round(latency_ms, 2),
                "mode": mode,
                "hop_depth": hop_depth,
                "retrieved_chunks": pipeline_result["retrieved_chunks"],
                "subgraph": pipeline_result["subgraph"],
                "prompt": pipeline_result["prompt"],
                "answer_model": answer_result.model_name,
            }
        )

    n = len(outputs) or 1
    summary = {
        "mode": mode,
        "hop_depth": hop_depth,
        "n_samples": len(outputs),
        "EM": round(sum(em_scores) / n, 4),
        "F1": round(sum(f1_scores) / n, 4),
        "latency_ms": round(sum(latencies) / n, 2),
        "recall_at_k": round(recall_hits / n, 4) if mode in ("rag", "graphrag") else None,
        "answer_in_subgraph": round(graph_hits / n, 4) if mode == "graphrag" else None,
    }
    return outputs, summary


def run_experiment(args) -> dict:
    rng = random.Random(args.seed) if args.seed is not None else random.Random()

    print("\nLoading KB...")
    triples, adjacency = load_kb(args.kb_path)
    corpus = triples_to_corpus(triples)

    print("\nLoading FAISS index...")
    faiss_index, corpus = load_or_build_index(
        corpus,
        index_path=args.index_path,
        corpus_path=args.corpus_path,
        force_rebuild=args.force_rebuild_index,
    )

    print("\nInitializing answer generator...")
    answer_generator = build_answer_generator(
        backend=args.backend,
        model_name=args.model_name,
        api_key=args.api_key,
    )

    results = {
        "metadata": {
            "split": args.split,
            "samples_per_hop": args.samples_per_hop,
            "k": args.k,
            "max_triples": args.max_triples,
            "backend": args.backend,
            "model_name": args.model_name,
            "seed": args.seed,
            "modes": args.modes,
        },
        "summaries": [],
        "outputs": [],
    }

    for hop_depth in (1, 2, 3):
        hop_dir = Path(args.data_dir) / f"{hop_depth}hop"
        splits = load_all_splits(str(hop_dir))
        sampled_pairs = sample_qa_pairs(
            splits[args.split],
            n_samples=args.samples_per_hop,
            rng=rng,
        )

        for mode in args.modes:
            outputs, summary = evaluate_mode_on_samples(
                qa_pairs=sampled_pairs,
                mode=mode,
                hop_depth=hop_depth,
                adjacency=adjacency,
                faiss_index=faiss_index,
                corpus=corpus,
                answer_generator=answer_generator,
                k=args.k,
                max_triples=args.max_triples,
            )
            results["summaries"].append(summary)
            results["outputs"].extend(outputs)

            print(
                f"{mode} | {hop_depth}-hop | "
                f"EM={summary['EM']:.4f} | F1={summary['F1']:.4f} | "
                f"latency={summary['latency_ms']:.2f}ms"
            )

    output_path = Path(args.output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    print(f"\nSaved results to {output_path}")
    return results


def parse_args():
    parser = argparse.ArgumentParser(description="Run Stage 2 RAG and Stage 3 GraphRAG.")
    parser.add_argument("--data_dir", default="data/raw")
    parser.add_argument("--kb_path", default="data/raw/kb.txt")
    parser.add_argument("--index_path", default="data/faiss/index.bin")
    parser.add_argument("--corpus_path", default="data/faiss/corpus.pkl")
    parser.add_argument("--output_path", default="data/processed/teacher_outputs/rag_graphrag_results.json")
    parser.add_argument("--split", default="test", choices=["train", "dev", "test"])
    parser.add_argument("--samples_per_hop", type=int, default=10)
    parser.add_argument("--k", type=int, default=5)
    parser.add_argument("--max_triples", type=int, default=50)
    parser.add_argument("--modes", nargs="+", default=["rag", "graphrag"], choices=["rag", "graphrag"])
    parser.add_argument("--backend", default="openai")
    parser.add_argument("--model_name", default="gpt-4o-mini")
    parser.add_argument("--api_key", default=None)
    parser.add_argument("--seed", type=int, default=None, help="Optional seed for reproducible random sampling.")
    parser.add_argument("--force_rebuild_index", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    run_experiment(parse_args())
