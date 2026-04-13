"""
src/data/inspect.py

Run full EDA on MetaQA dataset.
Answers:
  - How many 1-hop / 2-hop / 3-hop questions?
  - How big is the dataset?
  - Distribution of answer counts per question
  - Most common topic entities
  - Sample triples and QA pairs

Usage:
    python src/data/inspect.py --data_dir data/raw
"""

import argparse
import json
from collections import Counter
from pathlib import Path

from load_kb import load_kb, kb_stats
from load_metaqa import load_all_splits


# ─────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────

def section(title: str):
    width = 60
    print(f"\n{'=' * width}")
    print(f"  {title}")
    print(f"{'=' * width}")


def subsection(title: str):
    print(f"\n--- {title} ---")


# ─────────────────────────────────────────────
# EDA functions
# ─────────────────────────────────────────────

def eda_kb(kb_path: str):
    section("KNOWLEDGE BASE (kb.txt)")
    triples, adjacency = load_kb(kb_path)
    stats = kb_stats(triples, adjacency)

    print(f"  Total triples         : {stats['num_triples']:,}")
    print(f"  Unique entities       : {stats['num_entities']:,}")
    print(f"  Unique relations      : {stats['num_relations']:,}")
    print(f"  Nodes with outgoing   : {stats['num_subject_nodes']:,}")
    print(f"  Avg out-degree        : {stats['avg_out_degree']:.2f}")

    # Relation distribution
    subsection("Relation counts")
    rel_counter = Counter(r for _, r, _ in triples)
    for rel, count in rel_counter.most_common(10):
        print(f"    {rel:<35} {count:>6,}")

    # Degree distribution
    subsection("Out-degree distribution (top 10 entities)")
    degree_counter = Counter({k: len(v) for k, v in adjacency.items()})
    for entity, deg in degree_counter.most_common(10):
        print(f"    {entity:<35} {deg:>4} neighbours")

    subsection("Sample triples (first 5)")
    for t in triples[:5]:
        print(f"    {t[0]} | {t[1]} | {t[2]}")

    return triples, adjacency


def eda_hop(hop_name: str, hop_dir: str, n_samples: int = 10):
    section(f"QA PAIRS — {hop_name}")
    splits = load_all_splits(hop_dir)

    total = 0
    for split_name, pairs in splits.items():
        print(f"  {split_name:<8}: {len(pairs):>6,} questions")
        total += len(pairs)
    print(f"  {'TOTAL':<8}: {total:>6,} questions")

    # Answer count distribution
    subsection("Answer count distribution (train)")
    train = splits['train']
    answer_counts = Counter(len(p['answers']) for p in train)
    for count, freq in sorted(answer_counts.items()):
        bar = '█' * min(freq // max(1, len(train) // 50), 30)
        print(f"    {count:>2} answer(s): {freq:>6,}  {bar}")

    # Topic entity distribution
    subsection(f"Top 10 topic entities (train)")
    entity_counter = Counter(
        p['topic_entity'] for p in train if p['topic_entity']
    )
    for entity, count in entity_counter.most_common(10):
        print(f"    {entity:<35} {count:>5,}")

    # Missing entity bracket check
    missing = sum(1 for p in train if p['topic_entity'] is None)
    print(f"\n  Questions missing [entity] bracket: {missing}")

    # Sample QA pairs
    subsection(f"Sample QA pairs (first {n_samples})")
    for p in train[:n_samples]:
        answers_str = ' | '.join(p['answers'][:3])
        if len(p['answers']) > 3:
            answers_str += f" ... (+{len(p['answers'])-3} more)"
        print(f"    Q: {p['question']}")
        print(f"    A: {answers_str}")
        print()

    return splits


def eda_summary(hop_data: dict[str, dict]):
    section("SUMMARY — QUESTION COUNTS BY HOP")

    hop_totals = {}
    for hop_name, splits in hop_data.items():
        total = sum(len(s) for s in splits.values())
        hop_totals[hop_name] = {s: len(pairs) for s, pairs in splits.items()}
        hop_totals[hop_name]['total'] = total

    header = f"{'Hop':<10} {'Train':>8} {'Dev':>8} {'Test':>8} {'Total':>8}"
    print(f"  {header}")
    print(f"  {'-'*46}")
    grand_total = 0
    for hop_name, counts in hop_totals.items():
        print(
            f"  {hop_name:<10} "
            f"{counts.get('train', 0):>8,} "
            f"{counts.get('dev', 0):>8,} "
            f"{counts.get('test', 0):>8,} "
            f"{counts['total']:>8,}"
        )
        grand_total += counts['total']
    print(f"  {'-'*46}")
    print(f"  {'TOTAL':<10} {grand_total:>36,}")
    print()


# ─────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────

def main(data_dir: str, save_report: bool = False):
    data_dir = Path(data_dir)
    kb_path = data_dir / "kb.txt"

    # KB
    triples, adjacency = eda_kb(str(kb_path))

    # Hops
    hop_dirs = {
        "1-hop": str(data_dir / "1hop"),
        "2-hop": str(data_dir / "2hop"),
        "3-hop": str(data_dir / "3hop"),
    }

    hop_data = {}
    for hop_name, hop_dir in hop_dirs.items():
        if Path(hop_dir).exists():
            hop_data[hop_name] = eda_hop(hop_name, hop_dir, n_samples=5)
        else:
            print(f"\n[WARN] {hop_dir} not found, skipping.")

    eda_summary(hop_data)

    if save_report:
        report = {
            'kb': {
                'num_triples': len(triples),
                'num_entities': len(set(e for t in triples for e in (t[0], t[2]))),
            },
            'hops': {
                hop_name: {
                    split: len(pairs)
                    for split, pairs in splits.items()
                }
                for hop_name, splits in hop_data.items()
            }
        }
        out_path = Path("data/processed/eda_report.json")
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, 'w') as f:
            json.dump(report, f, indent=2)
        print(f"\nReport saved to {out_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="MetaQA EDA")
    parser.add_argument(
        "--data_dir", default="data/raw",
        help="Root directory containing kb.txt and 1hop/, 2hop/, 3hop/"
    )
    parser.add_argument(
        "--save_report", action="store_true",
        help="Save EDA stats to data/processed/eda_report.json"
    )
    args = parser.parse_args()
    main(args.data_dir, args.save_report)
