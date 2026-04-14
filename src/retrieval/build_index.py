"""
src/retrieval/build_index.py

One-time script to build the FAISS index over KB triples.
Run this ONCE before any RAG/GraphRAG experiments.

Usage:
    python src/retrieval/build_index.py --kb_path data/raw/kb.txt
"""

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "data"))
sys.path.insert(0, str(Path(__file__).parent))

from load_kb import load_kb
from faiss_index import triples_to_corpus, build_index


def main(args):
    print("="*50)
    print("Building FAISS index over KB triples")
    print("="*50)

    print(f"\nLoading KB from {args.kb_path}...")
    triples, _ = load_kb(args.kb_path)
    print(f"  {len(triples):,} triples loaded")

    print("\nConverting triples to corpus...")
    corpus = triples_to_corpus(triples)
    print(f"  {len(corpus):,} unique sentences")
    print(f"  Sample: {corpus[0]}")
    print(f"  Sample: {corpus[1]}")

    print(f"\nBuilding FAISS index...")
    t0 = time.time()
    index, corpus = build_index(
        corpus,
        index_path=args.index_path,
        corpus_path=args.corpus_path,
    )
    elapsed = time.time() - t0

    print(f"\nDone in {elapsed:.1f}s")
    print(f"Index: {index.ntotal:,} vectors")
    print(f"Saved to: {args.index_path}")
    print(f"Corpus saved to: {args.corpus_path}")
    print("\nRun `python src/retrieval/retrieve.py` to test retrieval.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--kb_path",     default="data/raw/kb.txt")
    parser.add_argument("--index_path",  default="data/faiss/index.bin")
    parser.add_argument("--corpus_path", default="data/faiss/corpus.pkl")
    args = parser.parse_args()
    main(args)
