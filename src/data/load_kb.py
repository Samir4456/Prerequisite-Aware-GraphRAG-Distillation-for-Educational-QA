"""
src/data/load_kb.py
Parse MetaQA knowledge base (kb.txt).
Each line format: subject|relation|object

Builds a BIDIRECTIONAL adjacency dict so graph traversal works
in both directions — critical for 1-hop and 2-hop questions where
the topic entity appears as the object, not the subject.
"""

from pathlib import Path
from collections import defaultdict


def load_kb(path: str) -> tuple[list[tuple], dict]:
    """
    Load the MetaQA knowledge graph from kb.txt.

    Returns:
        triples:    List of (subject, relation, object) tuples
        adjacency:  Bidirectional dict {entity -> [(relation, neighbour), ...]}
                    Includes both forward edges AND reverse edges (rel_reverse)
                    so traversal from any entity reaches its neighbours.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"KB file not found: {path}")

    triples = []
    adjacency = defaultdict(list)

    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split('|')
            if len(parts) != 3:
                continue
            subj, rel, obj = parts
            triples.append((subj, rel, obj))

            # Forward edge: subject → object
            adjacency[subj].append((rel, obj))

            # Reverse edge: object → subject
            # Prefix with "inv_" so serialised context is still readable
            adjacency[obj].append((f"inv_{rel}", subj))

    return triples, dict(adjacency)


def kb_stats(triples: list[tuple], adjacency: dict) -> dict:
    """Compute basic statistics on the knowledge base."""
    entities = set()
    relations = set()
    for subj, rel, obj in triples:
        entities.add(subj)
        entities.add(obj)
        relations.add(rel)

    return {
        'num_triples': len(triples),
        'num_entities': len(entities),
        'num_relations': len(relations),
        'num_nodes_in_adjacency': len(adjacency),
        'avg_degree': sum(len(v) for v in adjacency.values()) / len(adjacency) if adjacency else 0,
    }


if __name__ == "__main__":
    import sys
    path = sys.argv[1] if len(sys.argv) > 1 else "data/raw/kb.txt"
    triples, adjacency = load_kb(path)
    stats = kb_stats(triples, adjacency)
    print("=== Knowledge Base Stats ===")
    for k, v in stats.items():
        print(f"  {k}: {v:.1f}" if isinstance(v, float) else f"  {k}: {v}")
    print(f"\nSample triple: {triples[0]}")
    sample_entity = list(adjacency.keys())[0]
    print(f"Sample adjacency['{sample_entity}']: {adjacency[sample_entity][:3]}")