"""
src/data/load_metaqa.py
Load QA pairs from MetaQA hop files.
Each line format: question\tanswer1|answer2|...
"""

import re
from pathlib import Path


def extract_topic_entity(question: str) -> str | None:
    """MetaQA brackets topic entities like [Tom Hanks]."""
    match = re.search(r'\[(.+?)\]', question)
    return match.group(1) if match else None


def load_qa_pairs(path: str, max_samples: int = None) -> list[dict]:
    """
    Load QA pairs from a MetaQA file.

    Args:
        path: Path to qa_train.txt / qa_dev.txt / qa_test.txt
        max_samples: If set, return only the first N samples

    Returns:
        List of dicts: {question, answers, topic_entity}
    """
    pairs = []
    path = Path(path)

    if not path.exists():
        raise FileNotFoundError(f"QA file not found: {path}")

    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split('\t')
            if len(parts) != 2:
                continue
            question, answers_raw = parts
            pairs.append({
                'question': question,
                'answers': answers_raw.split('|'),
                'topic_entity': extract_topic_entity(question),
            })

    if max_samples:
        pairs = pairs[:max_samples]

    return pairs


def load_all_splits(hop_dir: str, max_samples: int = None) -> dict:
    """
    Load train / dev / test splits for a given hop directory.

    Args:
        hop_dir: e.g. 'data/raw/1hop'
        max_samples: limit per split (useful for quick dev)

    Returns:
        {'train': [...], 'dev': [...], 'test': [...]}
    """
    hop_dir = Path(hop_dir)
    splits = {}
    for split in ('train', 'dev', 'test'):
        fpath = hop_dir / f"qa_{split}.txt"
        splits[split] = load_qa_pairs(str(fpath), max_samples=max_samples)
    return splits


if __name__ == "__main__":
    # Quick sanity check
    import sys
    path = sys.argv[1] if len(sys.argv) > 1 else "data/raw/1hop/qa_train.txt"
    pairs = load_qa_pairs(path, max_samples=5)
    for p in pairs:
        print(p)
