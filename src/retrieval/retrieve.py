"""
src/retrieval/retrieve.py

Full retrieval pipeline:
  question → entity extraction → subgraph + FAISS → prompt

This is the core of Stage 2 (RAG) and Stage 3 (GraphRAG).
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "data"))
sys.path.insert(0, str(Path(__file__).parent.parent / "graph"))
sys.path.insert(0, str(Path(__file__).parent))

from entity_extract import extract_topic_entity, clean_question
from subgraph import get_subgraph, answer_in_subgraph
from serialize import serialize_triples, build_rag_prompt
from faiss_index import retrieve as faiss_retrieve
from embedder import embed_single


def run_rag_pipeline(
    question: str,
    adjacency: dict,
    faiss_index,
    corpus: list[str],
    hops: int = 1,
    k: int = 5,
    mode: str = "graphrag",
    max_triples: int = 50,
) -> dict:
    """
    Full RAG/GraphRAG pipeline for a single question.

    Args:
        question: Raw MetaQA question (with [brackets])
        adjacency: Bidirectional KB adjacency dict
        faiss_index: Loaded FAISS index
        corpus: Corpus list (same order as FAISS index)
        hops: Graph traversal depth (1, 2, or 3)
        k: Number of chunks to retrieve from FAISS
        mode: 'rag', 'graph', or 'graphrag'
        max_triples: Cap on subgraph size

    Returns:
        Dict with all pipeline outputs:
        {
            question, clean_question, topic_entity,
            subgraph, subgraph_text,
            retrieved_chunks, prompt
        }
    """
    # Step 1 — Extract entity
    topic_entity = extract_topic_entity(question)
    clean_q = clean_question(question)

    # Step 2 — Graph traversal
    subgraph = []
    if topic_entity and mode in ("graph", "graphrag"):
        subgraph = get_subgraph(
            topic_entity, adjacency, hops=hops, max_triples=max_triples
        )
    subgraph_text = serialize_triples(subgraph, style="arrow")

    # Step 3 — FAISS retrieval
    chunks = []
    if mode in ("rag", "graphrag"):
        chunks = faiss_retrieve(question, faiss_index, corpus, k=k)

    # Step 4 — Build prompt
    prompt = build_rag_prompt(
        clean_q, chunks, subgraph, mode=mode
    )

    return {
        "question": question,
        "clean_question": clean_q,
        "topic_entity": topic_entity,
        "subgraph": subgraph,
        "subgraph_text": subgraph_text,
        "retrieved_chunks": chunks,
        "prompt": prompt,
        "hop_depth": hops,
        "mode": mode,
    }


def recall_at_k(
    gold_answers: list[str],
    retrieved_chunks: list[str],
) -> bool:
    """
    Check if any gold answer appears in any retrieved chunk.
    Used to evaluate retrieval quality (Recall@K).
    """
    combined = " ".join(retrieved_chunks).lower()
    return any(ans.lower() in combined for ans in gold_answers)


def evaluate_retrieval(
    qa_pairs: list[dict],
    adjacency: dict,
    faiss_index,
    corpus: list[str],
    hops: int = 1,
    k: int = 5,
    n_samples: int = 200,
) -> dict:
    """
    Evaluate retrieval quality on a sample of QA pairs.
    Reports Recall@K and answer-in-subgraph rate.
    """
    from tqdm import tqdm

    pairs = qa_pairs[:n_samples]
    recall_hits = 0
    graph_hits = 0

    for item in tqdm(pairs, desc=f"Evaluating retrieval (k={k}, hops={hops})"):
        result = run_rag_pipeline(
            item['question'], adjacency, faiss_index, corpus,
            hops=hops, k=k, mode="graphrag"
        )
        if recall_at_k(item['answers'], result['retrieved_chunks']):
            recall_hits += 1
        if answer_in_subgraph(item['answers'], result['subgraph']):
            graph_hits += 1

    n = len(pairs)
    return {
        f'recall_at_{k}': round(recall_hits / n, 4),
        'answer_in_subgraph': round(graph_hits / n, 4),
        'n_samples': n,
    }


if __name__ == "__main__":
    import sys
    sys.path.insert(0, str(Path(__file__).parent.parent / "data"))
    from load_kb import load_kb
    from load_metaqa import load_all_splits
    from faiss_index import load_or_build_index, triples_to_corpus

    print("Loading KB...")
    triples, adjacency = load_kb("data/raw/kb.txt")
    corpus = triples_to_corpus(triples)

    print("Loading FAISS index...")
    index, corpus = load_or_build_index(corpus)

    print("Loading QA pairs...")
    splits = load_all_splits("data/raw/1hop", max_samples=5)

    print("\n=== Sample pipeline outputs ===")
    for item in splits['test']:
        result = run_rag_pipeline(
            item['question'], adjacency, index, corpus,
            hops=1, k=5, mode="graphrag"
        )
        print(f"\nQ: {result['question']}")
        print(f"Entity: {result['topic_entity']}")
        print(f"Subgraph ({len(result['subgraph'])} triples):")
        for t in result['subgraph'][:3]:
            print(f"  {t[0]} → {t[1]} → {t[2]}")
        print(f"Retrieved chunks: {len(result['retrieved_chunks'])}")
        for c in result['retrieved_chunks'][:2]:
            print(f"  - {c}")
        print(f"\nPrompt preview:\n{result['prompt'][:300]}...")
        print(f"\nGold answers: {item['answers']}")
        print("-" * 60)
