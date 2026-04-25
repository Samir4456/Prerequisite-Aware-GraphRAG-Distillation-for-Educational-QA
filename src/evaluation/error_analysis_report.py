"""
Generate report-ready error-source analysis tables and charts.

This script uses committed artifacts when full checkpoints/raw data are not
available. It produces CSV/JSON summaries and PNG figures under
results/error_analysis/ for direct use in reports and slide decks.

Run:
    python src/evaluation/error_analysis_report.py
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src" / "evaluation"))

from failure_modes import (  # noqa: E402
    analyze_case,
    answer_metrics,
    extract_final_answer,
    parse_answer_list,
    parse_context_sections,
)


PUBLISHED_MODEL_ROWS = [
    {
        "model": "DistilBERT baseline",
        "params": "66M",
        "family": "baseline",
        "training": "extractive",
        "retrieval": "reader context",
        "source": "README table",
        "sample_note": "reported benchmark",
        "1hop_EM": 0.4485,
        "2hop_EM": 0.6490,
        "3hop_EM": 0.0590,
        "overall_EM": None,
    },
    {
        "model": "Qwen2.5-3B Instruct base",
        "params": "3B",
        "family": "base",
        "training": "none",
        "retrieval": "GraphRAG",
        "source": "eval_results.json",
        "sample_note": "n=200 per hop",
        "results_path": "results/qwen2.5-eval/eval_results.json",
    },
    {
        "model": "Qwen2.5-1.5B RAG only Gold",
        "params": "1.5B",
        "family": "student",
        "training": "gold",
        "retrieval": "RAG only",
        "source": "README table",
        "sample_note": "reported benchmark",
        "1hop_EM": 0.720,
        "2hop_EM": 0.025,
        "3hop_EM": 0.020,
        "overall_EM": 0.255,
    },
    {
        "model": "Qwen2.5-0.5B GraphRAG Gold",
        "params": "0.5B",
        "family": "student",
        "training": "gold",
        "retrieval": "GraphRAG",
        "source": "README table",
        "sample_note": "reported benchmark",
        "1hop_EM": 0.756,
        "2hop_EM": 0.478,
        "3hop_EM": 0.056,
        "overall_EM": 0.430,
    },
    {
        "model": "Qwen2.5-1.5B GraphRAG Gold",
        "params": "1.5B",
        "family": "student",
        "training": "gold",
        "retrieval": "GraphRAG",
        "source": "README table",
        "sample_note": "reported benchmark",
        "1hop_EM": 0.778,
        "2hop_EM": 0.544,
        "3hop_EM": 0.054,
        "overall_EM": 0.459,
    },
    {
        "model": "Qwen2.5-3B GraphRAG Gold",
        "params": "3B",
        "family": "student",
        "training": "gold",
        "retrieval": "GraphRAG",
        "source": "README table",
        "sample_note": "reported benchmark",
        "1hop_EM": 0.832,
        "2hop_EM": 0.586,
        "3hop_EM": 0.050,
        "overall_EM": 0.489,
    },
    {
        "model": "Qwen2.5-0.5B GraphRAG Hybrid",
        "params": "0.5B",
        "family": "student",
        "training": "hybrid",
        "retrieval": "GraphRAG",
        "source": "eval_results.json",
        "sample_note": "n=500 per hop",
        "results_path": "results/qwen2.5-0.5b-graphrag-hybrid/eval_results.json",
        "examples_path": "results/qwen2.5-0.5b-graphrag-hybrid/eval_examples.json",
    },
    {
        "model": "Qwen2.5-1.5B GraphRAG Hybrid",
        "params": "1.5B",
        "family": "student",
        "training": "hybrid",
        "retrieval": "GraphRAG",
        "source": "eval_results.json",
        "sample_note": "n=500 per hop",
        "results_path": "results/qwen2.5-1.5b-graphrag-hybrid/eval_results.json",
        "examples_path": "results/qwen2.5-1.5b-graphrag-hybrid/eval_examples.json",
    },
    {
        "model": "Qwen2.5-3B GraphRAG Hybrid",
        "params": "3B",
        "family": "student",
        "training": "hybrid",
        "retrieval": "GraphRAG",
        "source": "eval_results.json",
        "sample_note": "n=500 per hop",
        "results_path": "results/qwen2.5-3b-graphrag-hybrid/eval_results.json",
        "examples_path": "results/qwen2.5-3b-graphrag-hybrid/eval_examples.json",
    },
]


DATASET_PATHS = {
    "GraphRAG Gold": ROOT / "data" / "processed" / "instruction_pairs" / "train_graphrag_gold_with_meta.json",
    "RAG Gold": ROOT / "data" / "processed" / "instruction_pairs" / "train_rag_gold_with_meta.json",
    "GraphRAG Hybrid": ROOT / "data" / "processed" / "instruction_pairs" / "train_graphrag_hybrid.json",
}


def read_text_any(path: Path) -> str:
    for encoding in ("utf-8", "cp1252", "latin-1"):
        try:
            return path.read_text(encoding=encoding)
        except UnicodeDecodeError:
            continue
    return path.read_text(encoding="utf-8", errors="replace")


def load_json_any(path: Path) -> Any | None:
    if not path.exists():
        return None
    text = read_text_any(path)
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        return None


def salvage_json_objects(path: Path) -> list[dict]:
    """Best-effort recovery for truncated eval_examples files."""
    if not path.exists():
        return []
    text = read_text_any(path)
    objects = []
    decoder = json.JSONDecoder()
    index = 0
    while True:
        start = text.find("{", index)
        if start == -1:
            break
        try:
            obj, end = decoder.raw_decode(text[start:])
        except json.JSONDecodeError:
            index = start + 1
            continue
        if isinstance(obj, dict) and "question" in obj and "gold" in obj:
            objects.append(obj)
        index = start + end
    return objects


def load_examples(path: Path) -> tuple[list[dict], str]:
    data = load_json_any(path)
    if isinstance(data, list):
        return data, "valid_json"
    salvaged = salvage_json_objects(path)
    return salvaged, "salvaged_partial_json" if salvaged else "unavailable"


def metrics_from_result_file(row: dict) -> dict:
    result_path = row.get("results_path")
    if not result_path:
        return row

    data = load_json_any(ROOT / result_path)
    if not data:
        return row

    for hop in (1, 2, 3):
        row[f"{hop}hop_EM"] = data.get(f"test_{hop}hop/EM")
        row[f"{hop}hop_F1"] = data.get(f"test_{hop}hop/F1")
        row[f"{hop}hop_latency_ms"] = data.get(f"test_{hop}hop/latency_ms")
    row["overall_EM"] = data.get("overall/EM", row.get("overall_EM"))
    row["overall_F1"] = data.get("overall/F1", row.get("overall_F1"))
    row["n_samples"] = data.get("n_samples")
    return row


def build_model_metrics() -> pd.DataFrame:
    rows = []
    for base_row in PUBLISHED_MODEL_ROWS:
        row = dict(base_row)
        rows.append(metrics_from_result_file(row))
    df = pd.DataFrame(rows)
    df["model_label"] = df["model"].apply(short_model_name)

    for hop in (1, 2, 3):
        if f"{hop}hop_EM" in df and f"{hop}hop_F1" in df:
            df[f"{hop}hop_F1_minus_EM"] = df[f"{hop}hop_F1"] - df[f"{hop}hop_EM"]
    return df


def short_model_name(model: str) -> str:
    replacements = {
        "DistilBERT baseline": "DistilBERT",
        "Qwen2.5-3B Instruct base": "Base 3B",
        "Qwen2.5-1.5B RAG only Gold": "RAG 1.5B",
        "Qwen2.5-0.5B GraphRAG Gold": "Gold 0.5B",
        "Qwen2.5-1.5B GraphRAG Gold": "Gold 1.5B",
        "Qwen2.5-3B GraphRAG Gold": "Gold 3B",
        "Qwen2.5-0.5B GraphRAG Hybrid": "Hybrid 0.5B",
        "Qwen2.5-1.5B GraphRAG Hybrid": "Hybrid 1.5B",
        "Qwen2.5-3B GraphRAG Hybrid": "Hybrid 3B",
    }
    return replacements.get(model, model)


def classify_answer_set_error(pred: list[str], gold: list[str]) -> dict:
    metrics = answer_metrics(pred, gold)
    if metrics.exact_match == 1.0:
        label = "correct"
    elif not pred:
        label = "empty output"
    elif metrics.f1 == 0:
        label = "exact miss"
    elif metrics.missing and metrics.extra:
        label = "partial: missing + extra"
    elif metrics.missing:
        label = "partial: incomplete answer set"
    elif metrics.extra:
        label = "partial: overgeneration"
    else:
        label = "formatting / set mismatch"

    return {
        "answer_set_error": label,
        "em": metrics.exact_match,
        "f1": metrics.f1,
        "precision": metrics.precision,
        "recall": metrics.recall,
        "missing_count": len(metrics.missing),
        "extra_count": len(metrics.extra),
        "gold_answer_count": len(gold),
        "pred_answer_count": len(pred),
    }


def build_answer_error_tables(model_df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    example_rows = []
    coverage_rows = []

    for _, model_row in model_df.iterrows():
        examples_path = model_row.get("examples_path")
        if not isinstance(examples_path, str):
            continue
        examples, load_status = load_examples(ROOT / examples_path)
        if not examples:
            continue

        for example_index, example in enumerate(examples):
            pred = example.get("pred") or parse_answer_list(example.get("raw_output", ""))
            gold = example.get("gold") or []
            metrics = classify_answer_set_error(pred, gold)
            sections = parse_context_sections(example.get("input_text", ""))
            context_lines = [
                *sections.get("graph_lines", []),
                *sections.get("retrieved_lines", []),
                *[str(chunk) for chunk in example.get("retrieved_chunks", [])],
            ]
            if context_lines:
                contextual = analyze_case(
                    gold_answers=gold,
                    raw_output=example.get("raw_output", ""),
                    predicted_answers=pred,
                    context_lines=context_lines,
                )
                failure_mode = contextual["failure_mode"]
                gold_any_in_context = contextual["coverage"]["gold"]["any"]
                gold_all_in_context = contextual["coverage"]["gold"]["all"]
                evidence_support_ratio = contextual["evidence_support"]["support_ratio"]
            else:
                failure_mode = metrics["answer_set_error"]
                gold_any_in_context = None
                gold_all_in_context = None
                evidence_support_ratio = None

            example_rows.append({
                "model": model_row["model"],
                "model_label": model_row.get("model_label", short_model_name(model_row["model"])),
                "params": model_row.get("params"),
                "training": model_row.get("training"),
                "example_index": example_index,
                "hop": example.get("hop"),
                "question": example.get("question"),
                "load_status": load_status,
                "context_available": bool(context_lines),
                "failure_mode": failure_mode,
                "gold_any_in_context": gold_any_in_context,
                "gold_all_in_context": gold_all_in_context,
                "evidence_support_ratio": evidence_support_ratio,
                **metrics,
            })

    examples_df = pd.DataFrame(example_rows)
    if examples_df.empty:
        return examples_df, pd.DataFrame()

    grouped = (
        examples_df
        .groupby(["model", "model_label", "answer_set_error"], dropna=False)
        .size()
        .reset_index(name="count")
    )
    totals = examples_df.groupby(["model", "model_label"]).size().rename("total").reset_index()
    summary = grouped.merge(totals, on=["model", "model_label"])
    summary["rate"] = summary["count"] / summary["total"]

    per_model = (
        examples_df
        .groupby(["model", "model_label"])
        .agg(
            saved_examples=("question", "count"),
            avg_example_f1=("f1", "mean"),
            partial_overlap_rate=("answer_set_error", lambda s: s.astype(str).str.startswith("partial").mean()),
            exact_miss_rate=("answer_set_error", lambda s: (s == "exact miss").mean()),
            avg_gold_answer_count=("gold_answer_count", "mean"),
            avg_pred_answer_count=("pred_answer_count", "mean"),
        )
        .reset_index()
    )
    coverage_rows = summary.merge(per_model, on=["model", "model_label"], how="left")
    return examples_df, coverage_rows


def analyze_dataset_item(dataset_name: str, item: dict) -> dict:
    metadata = item.get("metadata") or {}
    gold = metadata.get("gold_answers") or parse_answer_list(item.get("output", ""))
    sections = parse_context_sections(item.get("input", ""))
    graph_lines = sections["graph_lines"]
    retrieved_lines = sections["retrieved_lines"]
    context_lines = [*graph_lines, *retrieved_lines]
    output = item.get("output", "")
    predicted = parse_answer_list(extract_final_answer(output))

    analysis = analyze_case(
        gold_answers=gold,
        raw_output=output,
        predicted_answers=predicted,
        context_lines=context_lines,
    )

    graph_analysis = analyze_case(
        gold_answers=gold,
        raw_output=output,
        predicted_answers=predicted,
        context_lines=graph_lines,
    )
    retrieval_analysis = analyze_case(
        gold_answers=gold,
        raw_output=output,
        predicted_answers=predicted,
        context_lines=retrieved_lines,
    )

    support_ratio = analysis["evidence_support"]["support_ratio"]
    evidence_count = analysis["evidence_support"]["total"]
    unsupported_count = len(analysis["evidence_support"]["unsupported"])

    return {
        "dataset": dataset_name,
        "hop": metadata.get("hop"),
        "mode": metadata.get("mode"),
        "label_source": metadata.get("label_source"),
        "question": metadata.get("question") or sections["question"],
        "gold_answer_count": len(gold),
        "output_answer_count": len(predicted),
        "final_answer_em": analysis["metrics"]["em"],
        "final_answer_f1": analysis["metrics"]["f1"],
        "gold_any_in_context": analysis["coverage"]["gold"]["any"],
        "gold_all_in_context": analysis["coverage"]["gold"]["all"],
        "gold_any_in_graph": graph_analysis["coverage"]["gold"]["any"],
        "gold_all_in_graph": graph_analysis["coverage"]["gold"]["all"],
        "gold_any_in_retrieval": retrieval_analysis["coverage"]["gold"]["any"],
        "gold_all_in_retrieval": retrieval_analysis["coverage"]["gold"]["all"],
        "evidence_count": evidence_count,
        "evidence_support_ratio": support_ratio,
        "unsupported_evidence_count": unsupported_count,
        "has_unsupported_evidence": unsupported_count > 0,
        "failure_mode": analysis["failure_mode"],
    }


def build_dataset_trace_tables() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    rows = []
    for dataset_name, path in DATASET_PATHS.items():
        data = load_json_any(path)
        if not isinstance(data, list):
            continue
        for item in data:
            rows.append(analyze_dataset_item(dataset_name, item))

    item_df = pd.DataFrame(rows)
    if item_df.empty:
        return item_df, pd.DataFrame(), pd.DataFrame()

    summary = (
        item_df
        .groupby(["dataset", "hop"], dropna=False)
        .agg(
            n=("question", "count"),
            avg_gold_answer_count=("gold_answer_count", "mean"),
            final_answer_em=("final_answer_em", "mean"),
            final_answer_f1=("final_answer_f1", "mean"),
            gold_all_in_context_rate=("gold_all_in_context", "mean"),
            gold_any_in_context_rate=("gold_any_in_context", "mean"),
            gold_all_in_graph_rate=("gold_all_in_graph", "mean"),
            gold_any_in_graph_rate=("gold_any_in_graph", "mean"),
            gold_all_in_retrieval_rate=("gold_all_in_retrieval", "mean"),
            gold_any_in_retrieval_rate=("gold_any_in_retrieval", "mean"),
            avg_evidence_count=("evidence_count", "mean"),
            avg_evidence_support_ratio=("evidence_support_ratio", "mean"),
            unsupported_evidence_rate=("has_unsupported_evidence", "mean"),
        )
        .reset_index()
    )

    failure_modes = (
        item_df
        .groupby(["dataset", "hop", "failure_mode"], dropna=False)
        .size()
        .reset_index(name="count")
    )
    totals = item_df.groupby(["dataset", "hop"], dropna=False).size().rename("total").reset_index()
    failure_modes = failure_modes.merge(totals, on=["dataset", "hop"], how="left")
    failure_modes["rate"] = failure_modes["count"] / failure_modes["total"]
    return item_df, summary, failure_modes


def ensure_style() -> None:
    plt.rcParams.update({
        "figure.facecolor": "#fbfaf6",
        "axes.facecolor": "#fbfaf6",
        "axes.edgecolor": "#30312d",
        "axes.labelcolor": "#30312d",
        "xtick.color": "#30312d",
        "ytick.color": "#30312d",
        "text.color": "#30312d",
        "font.size": 10,
        "axes.titleweight": "bold",
    })


def save_bar_chart(
    df: pd.DataFrame,
    x: str,
    y: str,
    hue: str | None,
    title: str,
    ylabel: str,
    path: Path,
    rotate: int = 25,
) -> None:
    ensure_style()
    fig, ax = plt.subplots(figsize=(12, 6.5))

    if hue:
        x_order = list(dict.fromkeys(df[x].tolist()))
        pivot = df.pivot(index=x, columns=hue, values=y).reindex(x_order)
        pivot.plot(kind="bar", ax=ax, width=0.82)
        ax.legend(title=hue, frameon=False, loc="best")
    else:
        df.plot(kind="bar", x=x, y=y, legend=False, ax=ax, width=0.75)

    ax.set_title(title)
    ax.set_ylabel(ylabel)
    ax.set_xlabel("")
    ax.grid(axis="y", alpha=0.22)
    ax.set_ylim(bottom=0)
    if df[y].dropna().max() <= 1:
        ax.set_ylim(0, 1.02)
    plt.xticks(rotation=rotate, ha="right")
    fig.tight_layout()
    fig.savefig(path, dpi=220)
    plt.close(fig)


def plot_model_em(model_df: pd.DataFrame, out_dir: Path) -> None:
    rows = []
    for _, row in model_df.iterrows():
        for hop in (1, 2, 3):
            value = row.get(f"{hop}hop_EM")
            if pd.notna(value):
                rows.append({
                    "model": row["model_label"],
                    "hop": f"{hop}-hop",
                    "EM": float(value),
                })
    plot_df = pd.DataFrame(rows)
    save_bar_chart(
        plot_df,
        x="model",
        y="EM",
        hue="hop",
        title="Exact Match by Hop Across Available Models",
        ylabel="Exact Match",
        path=out_dir / "model_em_by_hop.png",
    )


def plot_f1_em_gap(model_df: pd.DataFrame, out_dir: Path) -> None:
    rows = []
    for _, row in model_df.iterrows():
        for hop in (1, 2, 3):
            em = row.get(f"{hop}hop_EM")
            f1 = row.get(f"{hop}hop_F1")
            if pd.notna(em) and pd.notna(f1):
                rows.append({
                    "model": row["model_label"],
                    "hop": f"{hop}-hop",
                    "F1-EM gap": float(f1) - float(em),
                })
    if not rows:
        return
    plot_df = pd.DataFrame(rows)
    save_bar_chart(
        plot_df,
        x="model",
        y="F1-EM gap",
        hue="hop",
        title="F1 minus EM Gap: Partial-Credit Signal",
        ylabel="F1 - EM",
        path=out_dir / "f1_em_gap_by_hop.png",
    )


def plot_retrieval_coverage(summary_df: pd.DataFrame, out_dir: Path) -> None:
    rows = []
    for _, row in summary_df.iterrows():
        for metric, label in [
            ("gold_all_in_context_rate", "all gold answers in context"),
            ("gold_any_in_context_rate", "any gold answer in context"),
        ]:
            if pd.notna(row.get(metric)):
                rows.append({
                    "dataset_hop": f"{row['dataset']} {int(row['hop'])}-hop",
                    "coverage": label,
                    "rate": row[metric],
                })
    plot_df = pd.DataFrame(rows)
    save_bar_chart(
        plot_df,
        x="dataset_hop",
        y="rate",
        hue="coverage",
        title="Retrieval / Context Coverage by Dataset and Hop",
        ylabel="Coverage rate",
        path=out_dir / "retrieval_coverage_by_hop.png",
    )


def plot_trace_quality(summary_df: pd.DataFrame, out_dir: Path) -> None:
    hybrid = summary_df[summary_df["dataset"] == "GraphRAG Hybrid"].copy()
    if hybrid.empty:
        return
    rows = []
    for _, row in hybrid.iterrows():
        rows.extend([
            {
                "hop": f"{int(row['hop'])}-hop",
                "metric": "avg evidence support",
                "rate": row["avg_evidence_support_ratio"],
            },
            {
                "hop": f"{int(row['hop'])}-hop",
                "metric": "unsupported evidence rate",
                "rate": row["unsupported_evidence_rate"],
            },
        ])
    plot_df = pd.DataFrame(rows)
    save_bar_chart(
        plot_df,
        x="hop",
        y="rate",
        hue="metric",
        title="Teacher Evidence Trace Quality",
        ylabel="Rate",
        path=out_dir / "teacher_trace_quality.png",
        rotate=0,
    )


def plot_answer_burden(summary_df: pd.DataFrame, out_dir: Path) -> None:
    gold = summary_df[summary_df["dataset"] == "GraphRAG Gold"].copy()
    if gold.empty:
        gold = summary_df.drop_duplicates("hop").copy()
    gold["hop_label"] = gold["hop"].astype(int).astype(str) + "-hop"
    save_bar_chart(
        gold,
        x="hop_label",
        y="avg_gold_answer_count",
        hue=None,
        title="Hop-wise Gold Answer Count: Output Burden",
        ylabel="Average gold answers per question",
        path=out_dir / "answer_burden_by_hop.png",
        rotate=0,
    )


def plot_answer_set_errors(answer_summary_df: pd.DataFrame, out_dir: Path) -> None:
    if answer_summary_df.empty:
        return
    plot_df = answer_summary_df.copy()
    keep = [
        "partial: incomplete answer set",
        "partial: overgeneration",
        "partial: missing + extra",
        "exact miss",
        "empty output",
    ]
    plot_df = plot_df[plot_df["answer_set_error"].isin(keep)]
    if plot_df.empty:
        return
    save_bar_chart(
        plot_df,
        x="model_label",
        y="rate",
        hue="answer_set_error",
        title="Saved Example Answer-Set Error Breakdown",
        ylabel="Rate among saved examples",
        path=out_dir / "answer_set_error_breakdown.png",
    )


def write_summary_markdown(
    out_dir: Path,
    model_df: pd.DataFrame,
    trace_summary_df: pd.DataFrame,
    answer_summary_df: pd.DataFrame,
) -> None:
    best_overall = (
        model_df.dropna(subset=["overall_EM"])
        .sort_values("overall_EM", ascending=False)
        .head(1)
    )
    best_name = best_overall.iloc[0]["model"] if not best_overall.empty else "n/a"
    best_em = best_overall.iloc[0]["overall_EM"] if not best_overall.empty else None

    base = model_df[model_df["model"] == "Qwen2.5-3B Instruct base"]
    hybrid = model_df[model_df["model"] == "Qwen2.5-3B GraphRAG Hybrid"]
    base_delta = None
    if not base.empty and not hybrid.empty:
        base_delta = float(hybrid.iloc[0]["overall_EM"]) - float(base.iloc[0]["overall_EM"])

    lines = [
        "# Error Source Analysis Summary",
        "",
        f"- Best available overall EM row: {best_name} ({best_em:.3f})." if best_em is not None else "- Best model: n/a.",
        "- Use Qwen2.5-3B GraphRAG Gold as the best answer-accuracy model in the report.",
        "- Use Qwen2.5-3B GraphRAG Hybrid for the evidence-trace demo, because it is trained with teacher traces.",
    ]
    if base_delta is not None:
        lines.append(
            f"- Same-size base vs trained signal: 3B Hybrid improves overall EM over 3B base by {base_delta:.3f}; sample sizes differ, so frame this as suggestive context-use evidence."
        )

    if not trace_summary_df.empty:
        hybrid_trace = trace_summary_df[trace_summary_df["dataset"] == "GraphRAG Hybrid"]
        if not hybrid_trace.empty:
            unsupported = hybrid_trace["unsupported_evidence_rate"].mean()
            support = hybrid_trace["avg_evidence_support_ratio"].mean()
            lines.append(
                f"- Teacher trace quality: average evidence support is {support:.3f}, with unsupported-evidence examples at {unsupported:.3f} across hops."
            )

    if not answer_summary_df.empty:
        partial_rows = answer_summary_df[
            answer_summary_df["answer_set_error"].astype(str).str.startswith("partial")
        ]
        partial = partial_rows.groupby("model")["rate"].sum().sort_values(ascending=False)
        if not partial.empty:
            lines.append(
                f"- Saved prediction examples show partial-overlap errors most strongly for {partial.index[0]} ({partial.iloc[0]:.3f} of saved examples)."
            )

    lines.extend([
        "",
        "## Visuals",
        "",
        "- `model_em_by_hop.png`: compare all available models by hop.",
        "- `f1_em_gap_by_hop.png`: shows where F1 is much higher than EM, indicating partial answer-set overlap.",
        "- `retrieval_coverage_by_hop.png`: gold answer coverage in GraphRAG/RAG contexts.",
        "- `teacher_trace_quality.png`: evidence support and unsupported-evidence rate.",
        "- `answer_burden_by_hop.png`: hop-wise multi-answer burden.",
        "- `answer_set_error_breakdown.png`: saved prediction error modes.",
        "",
        "## Framing",
        "",
        "Call this failure mode analysis or error source analysis, not absolute causal proof. Retrieval, evidence grounding, path selection, formatting, and supervision noise interact.",
    ])

    (out_dir / "summary.md").write_text("\n".join(lines), encoding="utf-8")


def write_json_summary(out_dir: Path, **tables: pd.DataFrame) -> None:
    summary = {}
    for name, df in tables.items():
        if df.empty:
            summary[name] = []
        else:
            summary[name] = json.loads(df.head(50).to_json(orient="records"))
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")


def run(args: argparse.Namespace) -> None:
    out_dir = ROOT / args.output_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    model_df = build_model_metrics()
    examples_df, answer_summary_df = build_answer_error_tables(model_df)
    trace_item_df, trace_summary_df, trace_failure_df = build_dataset_trace_tables()

    model_df.to_csv(out_dir / "model_metrics_all.csv", index=False)
    examples_df.to_csv(out_dir / "saved_example_answer_errors.csv", index=False)
    answer_summary_df.to_csv(out_dir / "answer_set_error_breakdown.csv", index=False)
    trace_item_df.to_csv(out_dir / "dataset_trace_items.csv", index=False)
    trace_summary_df.to_csv(out_dir / "dataset_trace_summary_by_hop.csv", index=False)
    trace_failure_df.to_csv(out_dir / "dataset_trace_failure_modes.csv", index=False)

    plot_model_em(model_df, out_dir)
    plot_f1_em_gap(model_df, out_dir)
    if not trace_summary_df.empty:
        plot_retrieval_coverage(trace_summary_df, out_dir)
        plot_trace_quality(trace_summary_df, out_dir)
        plot_answer_burden(trace_summary_df, out_dir)
    plot_answer_set_errors(answer_summary_df, out_dir)

    write_summary_markdown(out_dir, model_df, trace_summary_df, answer_summary_df)
    write_json_summary(
        out_dir,
        model_metrics=model_df,
        answer_set_errors=answer_summary_df,
        trace_summary=trace_summary_df,
        trace_failure_modes=trace_failure_df,
    )

    print(f"Error analysis outputs written to: {out_dir}")
    print(f"Models compared: {len(model_df)}")
    print(f"Saved examples analyzed: {len(examples_df)}")
    print(f"Dataset trace items analyzed: {len(trace_item_df)}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate error analysis tables and charts.")
    parser.add_argument("--output_dir", default="results/error_analysis")
    args = parser.parse_args()
    run(args)


if __name__ == "__main__":
    main()
