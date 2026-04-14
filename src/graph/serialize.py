"""
src/graph/serialize.py
Convert subgraph triples into text formats suitable for LLM prompts.
"""


def serialize_triples(triples: list[tuple], style: str = "arrow") -> str:
    """
    Convert list of (subject, relation, object) triples to a text string.

    Args:
        triples: List of (subj, rel, obj) tuples
        style: One of:
            'arrow'    → "The Matrix → directed_by → Wachowski Sisters"
            'sentence' → "The Matrix directed_by Wachowski Sisters."
            'natural'  → "The Matrix is directed_by Wachowski Sisters."

    Returns:
        Multi-line string of serialised triples
    """
    if not triples:
        return ""

    lines = []
    for subj, rel, obj in triples:
        if style == "arrow":
            lines.append(f"{subj} → {rel} → {obj}")
        elif style == "sentence":
            lines.append(f"{subj} {rel} {obj}.")
        elif style == "natural":
            lines.append(f"{subj} is {rel.replace('_', ' ')} {obj}.")
        else:
            raise ValueError(f"Unknown style: {style}")

    return "\n".join(lines)


def build_rag_prompt(
    question: str,
    retrieved_chunks: list[str],
    subgraph_triples: list[tuple] = None,
    style: str = "arrow",
    mode: str = "graphrag",   # "rag", "graph", or "graphrag"
) -> str:
    """
    Build the full prompt to send to GPT-4o (teacher model).

    Args:
        question: The cleaned question (no brackets)
        retrieved_chunks: Top-K strings from FAISS retrieval
        subgraph_triples: List of (subj, rel, obj) from graph traversal
        style: Serialisation style for graph triples
        mode: What context to include:
            'rag'      → retrieved chunks only
            'graph'    → subgraph only
            'graphrag' → both (default)

    Returns:
        Full prompt string ready to send to GPT-4o
    """
    sections = []
    sections.append(
        "You are a question answering assistant. "
        "Answer the question based on the provided context. "
        "Be concise — give only the answer entity or entities, "
        "separated by | if there are multiple."
    )

    if mode in ("graph", "graphrag") and subgraph_triples:
        graph_text = serialize_triples(subgraph_triples, style=style)
        sections.append(f"Knowledge Graph:\n{graph_text}")

    if mode in ("rag", "graphrag") and retrieved_chunks:
        context_text = "\n".join(f"- {c}" for c in retrieved_chunks)
        sections.append(f"Retrieved Context:\n{context_text}")

    sections.append(f"Question: {question}")
    sections.append("Answer:")

    return "\n\n".join(sections)


if __name__ == "__main__":
    triples = [
        ("The Matrix", "directed_by", "Wachowski Sisters"),
        ("The Matrix", "has_genre", "Sci-Fi"),
        ("The Matrix", "release_year", "1999"),
    ]
    chunks = [
        "The Matrix is a 1999 science fiction film directed by the Wachowski Sisters.",
        "Lana and Lilly Wachowski are American film directors known for The Matrix trilogy.",
    ]
    question = "Who directed The Matrix?"

    prompt = build_rag_prompt(question, chunks, triples, mode="graphrag")
    print(prompt)
    print("\n" + "="*50)
    print("\nGraph only:")
    print(build_rag_prompt(question, chunks, triples, mode="graph"))
    print("\n" + "="*50)
    print("\nRAG only:")
    print(build_rag_prompt(question, chunks, triples, mode="rag"))
