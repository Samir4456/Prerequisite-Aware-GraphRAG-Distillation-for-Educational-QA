"""
src/graph/subgraph.py
N-hop subgraph extraction from the MetaQA knowledge base.
Uses bidirectional adjacency dict from load_kb.py.
"""


def get_subgraph(
    topic_entity: str,
    adjacency: dict,
    hops: int = 1,
    max_triples: int = 100,
) -> list[tuple]:
    """
    Extract N-hop subgraph starting from topic_entity.

    Args:
        topic_entity: Starting entity (e.g. "The Matrix")
        adjacency: Bidirectional adjacency dict from load_kb()
        hops: Number of hops to traverse (1, 2, or 3)
        max_triples: Cap on total triples returned (avoids huge 3-hop graphs)

    Returns:
        List of (subject, relation, object) triples
        Forward triples only — inv_ edges are excluded from output
        but used internally for traversal.
    """
    if not topic_entity or topic_entity not in adjacency:
        return []

    subgraph = []
    frontier = {topic_entity}
    visited = {topic_entity}

    for hop in range(hops):
        next_frontier = set()
        for entity in frontier:
            for rel, obj in adjacency.get(entity, []):
                # Add to subgraph (forward edges only for readability)
                if not rel.startswith("inv_"):
                    subgraph.append((entity, rel, obj))
                else:
                    # Reverse edge: entity is obj, obj is subj
                    real_subj = obj
                    real_rel = rel[4:]   # strip "inv_"
                    subgraph.append((real_subj, real_rel, entity))

                if obj not in visited:
                    next_frontier.add(obj)
                    visited.add(obj)

                if len(subgraph) >= max_triples:
                    return _deduplicate(subgraph)

        frontier = next_frontier

    return _deduplicate(subgraph)


def _deduplicate(triples: list[tuple]) -> list[tuple]:
    """Remove duplicate triples while preserving order."""
    seen = set()
    result = []
    for t in triples:
        if t not in seen:
            result.append(t)
            seen.add(t)
    return result


def get_subgraph_entities(subgraph: list[tuple]) -> set[str]:
    """Return all entity names that appear in the subgraph."""
    entities = set()
    for subj, rel, obj in subgraph:
        entities.add(subj)
        entities.add(obj)
    return entities


def answer_in_subgraph(answers: list[str], subgraph: list[tuple]) -> bool:
    """Check if any gold answer appears as an entity in the subgraph."""
    entities = get_subgraph_entities(subgraph)
    entities_lower = {e.lower() for e in entities}
    return any(a.lower() in entities_lower for a in answers)


if __name__ == "__main__":
    import sys
    sys.path.insert(0, str(__import__('pathlib').Path(__file__).parent.parent / "data"))
    from load_kb import load_kb

    triples, adjacency = load_kb("data/raw/kb.txt")

    entity = "The Matrix"
    for hops in [1, 2]:
        sg = get_subgraph(entity, adjacency, hops=hops, max_triples=50)
        print(f"\n{hops}-hop subgraph for '{entity}': {len(sg)} triples")
        for t in sg[:5]:
            print(f"  {t[0]} | {t[1]} | {t[2]}")
        print("  ...")
