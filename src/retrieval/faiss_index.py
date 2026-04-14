"""
src/retrieval/faiss_index.py
Build a FAISS flat index over the KB triple corpus and query it.
Index is cached to disk so it only needs to be built once.
"""

import pickle
import time
from pathlib import Path

import faiss
import numpy as np

from embedder import embed, embed_single


# ─────────────────────────────────────────────
# Build
# ─────────────────────────────────────────────

def build_index(
    texts: list[str],
    index_path: str = "data/faiss/index.bin",
    corpus_path: str = "data/faiss/corpus.pkl",
    batch_size: int = 256,
) -> tuple[faiss.Index, list[str]]:
    """
    Encode all texts and build a FAISS flat L2 index.
    Saves index and corpus to disk.

    Args:
        texts: List of strings to index (one per KB triple)
        index_path: Where to save the FAISS index
        corpus_path: Where to save the corpus list
        batch_size: Embedding batch size

    Returns:
        (index, texts)
    """
    Path(index_path).parent.mkdir(parents=True, exist_ok=True)

    print(f"  Encoding {len(texts):,} texts...")
    t0 = time.time()
    embeddings = embed(texts, batch_size=batch_size, show_progress=True)
    print(f"  Encoded in {time.time()-t0:.1f}s  shape={embeddings.shape}")

    dim = embeddings.shape[1]
    index = faiss.IndexFlatIP(dim)   # Inner product (= cosine sim after L2 norm)
    index.add(embeddings.astype(np.float32))
    print(f"  FAISS index built: {index.ntotal:,} vectors")

    # Save to disk
    faiss.write_index(index, str(index_path))
    with open(corpus_path, 'wb') as f:
        pickle.dump(texts, f)
    print(f"  Saved → {index_path}")
    print(f"  Saved → {corpus_path}")

    return index, texts


# ─────────────────────────────────────────────
# Load
# ─────────────────────────────────────────────

def load_index(
    index_path: str = "data/faiss/index.bin",
    corpus_path: str = "data/faiss/corpus.pkl",
) -> tuple[faiss.Index, list[str]]:
    """Load a previously saved FAISS index and corpus from disk."""
    index_path = Path(index_path)
    corpus_path = Path(corpus_path)

    if not index_path.exists() or not corpus_path.exists():
        raise FileNotFoundError(
            f"FAISS index not found at {index_path}. "
            "Run build_index() first."
        )

    print(f"  Loading FAISS index from {index_path}...")
    index = faiss.read_index(str(index_path))
    with open(corpus_path, 'rb') as f:
        texts = pickle.load(f)
    print(f"  Loaded {index.ntotal:,} vectors  |  {len(texts):,} texts")

    return index, texts


def load_or_build_index(
    texts: list[str],
    index_path: str = "data/faiss/index.bin",
    corpus_path: str = "data/faiss/corpus.pkl",
    force_rebuild: bool = False,
) -> tuple[faiss.Index, list[str]]:
    """
    Load index from disk if it exists, otherwise build and save it.
    Pass force_rebuild=True to always rebuild.
    """
    if not force_rebuild and Path(index_path).exists():
        return load_index(index_path, corpus_path)
    print("  Building new FAISS index...")
    return build_index(texts, index_path, corpus_path)


# ─────────────────────────────────────────────
# Query
# ─────────────────────────────────────────────

def retrieve(
    question: str,
    index: faiss.Index,
    texts: list[str],
    k: int = 5,
) -> list[str]:
    """
    Retrieve top-K most relevant texts for a question.

    Args:
        question: The question string
        index: FAISS index
        texts: Corpus list (same order as index)
        k: Number of results to return

    Returns:
        List of top-K text strings
    """
    q_vec = embed_single(question).astype(np.float32)
    scores, indices = index.search(q_vec, k)
    return [texts[i] for i in indices[0] if i < len(texts)]


def retrieve_with_scores(
    question: str,
    index: faiss.Index,
    texts: list[str],
    k: int = 5,
) -> list[tuple[str, float]]:
    """Same as retrieve() but returns (text, score) tuples."""
    q_vec = embed_single(question).astype(np.float32)
    scores, indices = index.search(q_vec, k)
    return [(texts[i], float(scores[0][j]))
            for j, i in enumerate(indices[0]) if i < len(texts)]


# ─────────────────────────────────────────────
# Build corpus from KB triples
# ─────────────────────────────────────────────

def triples_to_corpus(triples: list[tuple]) -> list[str]:
    """
    Convert KB triples to natural language sentences for indexing.
    Only uses forward triples (not inv_ reverse edges).

    Example: ('The Matrix', 'directed_by', 'Wachowski Sisters')
             → 'The Matrix directed_by Wachowski Sisters'
    """
    corpus = []
    seen = set()
    for subj, rel, obj in triples:
        if rel.startswith("inv_"):
            continue   # skip reverse edges — corpus uses forward only
        text = f"{subj} {rel} {obj}"
        if text not in seen:
            corpus.append(text)
            seen.add(text)
    return corpus


if __name__ == "__main__":
    import sys
    sys.path.insert(0, str(__import__('pathlib').Path(__file__).parent.parent / "data"))
    from load_kb import load_kb

    triples, _ = load_kb("data/raw/kb.txt")
    corpus = triples_to_corpus(triples)
    print(f"Corpus size: {len(corpus):,}")
    print(f"Sample: {corpus[:3]}")

    index, texts = load_or_build_index(corpus)

    q = "Who directed The Matrix?"
    results = retrieve_with_scores(q, index, texts, k=5)
    print(f"\nQuery: {q}")
    for text, score in results:
        print(f"  [{score:.3f}] {text}")
