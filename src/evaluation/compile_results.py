"""
src/evaluation/compile_results.py

Reads all evaluation result JSON files and prints a clean
comparison table across all models and hop levels.

Usage:
    python src/evaluation/compile_results.py

Optionally save to CSV:
    python src/evaluation/compile_results.py --save_csv results/comparison.csv
"""

import argparse
import json
import os
from pathlib import Path


# ─────────────────────────────────────────────
# All models to compare
# ─────────────────────────────────────────────

MODELS = [
    {
        "name": "DistilBERT baseline",
        "path": "checkpoints/distilbert-baseline/results.json",
        "type": "baseline",
        "params": "66M",
    },
    {
        "name": "Qwen2.5-0.5B GraphRAG",
        "path": "results/qwen2.5-0.5b-graphrag-gold/eval_results.json",
        "type": "student",
        "params": "0.5B",
    },
    {
        "name": "Qwen2.5-1.5B GraphRAG",
        "path": "results/qwen2.5-1.5b-graphrag-gold/eval_results.json",
        "type": "student",
        "params": "1.5B",
    },
    {
        "name": "Qwen2.5-3B GraphRAG",
        "path": "results/qwen2.5-3b-graphrag-gold/eval_results.json",
        "type": "student",
        "params": "3B",
    },
    {
        "name": "Qwen2.5-1.5B RAG only",
        "path": "results/qwen2.5-1.5b-rag-gold/eval_results.json",
        "type": "ablation",
        "params": "1.5B",
    },
]

# ─────────────────────────────────────────────
# Key mapping between result JSON fields
# ─────────────────────────────────────────────

# DistilBERT results.json uses different key names than evaluate_student.py
DISTILBERT_KEY_MAP = {
    "test_1hop/EM":         ["test_1hop/EM", "test_1hop_EM"],
    "test_1hop/F1":         ["test_1hop/F1", "test_1hop_F1"],
    "test_1hop/latency_ms": ["test_1hop/latency_ms", "test_1hop_latency_ms"],
    "test_2hop/EM":         ["test_2hop/EM", "test_2hop_EM"],
    "test_2hop/F1":         ["test_2hop/F1", "test_2hop_F1"],
    "test_2hop/latency_ms": ["test_2hop/latency_ms", "test_2hop_latency_ms"],
    "test_3hop/EM":         ["test_3hop/EM", "test_3hop_EM"],
    "test_3hop/F1":         ["test_3hop/F1", "test_3hop_F1"],
    "test_3hop/latency_ms": ["test_3hop/latency_ms", "test_3hop_latency_ms"],
}


def get_val(data: dict, keys: list):
    """Try multiple key names, return first match or None."""
    for k in keys:
        if k in data:
            return data[k]
    return None


def load_results(path: str) -> dict | None:
    p = Path(path)
    if not p.exists():
        return None
    with open(p) as f:
        return json.load(f)


def extract_metrics(data: dict) -> dict:
    """Normalise result dict into standard metric names."""
    metrics = {}
    for standard_key, candidates in DISTILBERT_KEY_MAP.items():
        val = get_val(data, candidates)
        metrics[standard_key] = val
    # Overall EM/F1
    metrics["overall/EM"] = get_val(data, ["overall/EM", "overall_EM"])
    metrics["overall/F1"] = get_val(data, ["overall/F1", "overall_F1"])
    return metrics


def fmt(val, pct=True):
    """Format a metric value for display."""
    if val is None:
        return "—"
    if pct:
        return f"{val:.3f}"
    return f"{val:.1f}"


def print_table(rows: list[dict], title: str):
    """Print a formatted ASCII table."""
    print(f"\n{'='*80}")
    print(f"  {title}")
    print(f"{'='*80}")

    header = f"{'Model':<30} {'Params':>6}  {'1-hop EM':>8} {'2-hop EM':>8} {'3-hop EM':>8} {'Avg EM':>7}"
    print(header)
    print("-" * 80)

    for r in rows:
        m = r["metrics"]
        avg_em = None
        ems = [m.get("test_1hop/EM"), m.get("test_2hop/EM"), m.get("test_3hop/EM")]
        valid = [e for e in ems if e is not None]
        if valid:
            avg_em = sum(valid) / len(valid)

        print(
            f"{r['name']:<30} {r['params']:>6}  "
            f"{fmt(m.get('test_1hop/EM')):>8} "
            f"{fmt(m.get('test_2hop/EM')):>8} "
            f"{fmt(m.get('test_3hop/EM')):>8} "
            f"{fmt(avg_em):>7}"
        )

    print("-" * 80)


def print_full_table(rows: list[dict]):
    """Print full table including F1 and latency."""
    print(f"\n{'='*110}")
    print("  Full results — EM / F1 / Latency per hop")
    print(f"{'='*110}")

    header = (
        f"{'Model':<30} {'Params':>6}  "
        f"{'1h-EM':>6} {'1h-F1':>6} {'1h-ms':>6}  "
        f"{'2h-EM':>6} {'2h-F1':>6} {'2h-ms':>6}  "
        f"{'3h-EM':>6} {'3h-F1':>6} {'3h-ms':>6}"
    )
    print(header)
    print("-" * 110)

    for r in rows:
        m = r["metrics"]
        print(
            f"{r['name']:<30} {r['params']:>6}  "
            f"{fmt(m.get('test_1hop/EM')):>6} "
            f"{fmt(m.get('test_1hop/F1')):>6} "
            f"{fmt(m.get('test_1hop/latency_ms'), pct=False):>6}  "
            f"{fmt(m.get('test_2hop/EM')):>6} "
            f"{fmt(m.get('test_2hop/F1')):>6} "
            f"{fmt(m.get('test_2hop/latency_ms'), pct=False):>6}  "
            f"{fmt(m.get('test_3hop/EM')):>6} "
            f"{fmt(m.get('test_3hop/F1')):>6} "
            f"{fmt(m.get('test_3hop/latency_ms'), pct=False):>6}"
        )

    print("-" * 110)


def save_csv(rows: list[dict], path: str):
    import csv
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "model", "params", "type",
        "1hop_EM", "1hop_F1", "1hop_latency_ms",
        "2hop_EM", "2hop_F1", "2hop_latency_ms",
        "3hop_EM", "3hop_F1", "3hop_latency_ms",
        "overall_EM", "overall_F1",
    ]
    with open(path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for r in rows:
            m = r["metrics"]
            writer.writerow({
                "model":            r["name"],
                "params":           r["params"],
                "type":             r["type"],
                "1hop_EM":          m.get("test_1hop/EM", ""),
                "1hop_F1":          m.get("test_1hop/F1", ""),
                "1hop_latency_ms":  m.get("test_1hop/latency_ms", ""),
                "2hop_EM":          m.get("test_2hop/EM", ""),
                "2hop_F1":          m.get("test_2hop/F1", ""),
                "2hop_latency_ms":  m.get("test_2hop/latency_ms", ""),
                "3hop_EM":          m.get("test_3hop/EM", ""),
                "3hop_F1":          m.get("test_3hop/F1", ""),
                "3hop_latency_ms":  m.get("test_3hop/latency_ms", ""),
                "overall_EM":       m.get("overall/EM", ""),
                "overall_F1":       m.get("overall/F1", ""),
            })
    print(f"\nCSV saved → {path}")


def main(args):
    rows = []
    missing = []

    for model in MODELS:
        data = load_results(model["path"])
        if data is None:
            missing.append(model["name"])
            print(f"  [SKIP] {model['name']} — {model['path']} not found")
            continue

        metrics = extract_metrics(data)
        rows.append({
            "name":    model["name"],
            "params":  model["params"],
            "type":    model["type"],
            "metrics": metrics,
        })

    if not rows:
        print("\nNo result files found. Run evaluate_student.py first.")
        return

    # Split into groups
    baseline_rows  = [r for r in rows if r["type"] == "baseline"]
    student_rows   = [r for r in rows if r["type"] == "student"]
    ablation_rows  = [r for r in rows if r["type"] == "ablation"]

    # Print tables
    if baseline_rows:
        print_table(baseline_rows, "Baseline")
    if student_rows:
        print_table(student_rows, "Model size ablation — GraphRAG gold (0.5B / 1.5B / 3B)")
    if ablation_rows:
        print_table(ablation_rows, "Retrieval ablation — RAG only vs GraphRAG")

    # Full table with F1 and latency
    print_full_table(rows)

    # Summary: best model per hop
    print(f"\n{'='*80}")
    print("  Best model per hop (by Exact Match)")
    print(f"{'='*80}")
    for hop in [1, 2, 3]:
        key = f"test_{hop}hop/EM"
        best = max(rows, key=lambda r: r["metrics"].get(key) or 0)
        val = best["metrics"].get(key)
        print(f"  {hop}-hop: {best['name']:<30} EM={fmt(val)}")

    if missing:
        print(f"\nMissing result files ({len(missing)}):")
        for m in missing:
            print(f"  - {m}")
        print("Run evaluate_student.py for each missing model first.")

    if args.save_csv:
        save_csv(rows, args.save_csv)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Compile all model results into a table")
    parser.add_argument("--save_csv", default="results/comparison.csv",
                        help="Path to save CSV output")
    args = parser.parse_args()
    main(args)
