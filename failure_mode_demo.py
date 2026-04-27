"""
Streamlit MVP demo for GraphRAG failure-mode analysis.

Run:
    streamlit run failure_mode_demo.py
"""

from __future__ import annotations

import json
import re
import sys
import time
from pathlib import Path

import pandas as pd
import streamlit as st
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src" / "evaluation"))

from failure_modes import (  # noqa: E402
    analyze_case,
    answer_metrics,
    normalize_text,
    normalize_answer,
    parse_context_sections,
)
from error_analysis_report import answer_coverage_in_lines, load_examples, short_model_name  # noqa: E402


TRACE_PATH = ROOT / "data" / "processed" / "instruction_pairs" / "train_graphrag_hybrid.json"
LOCAL_3B_HYBRID_RESULTS = ROOT / "results" / "qwen2.5-3b-graphrag-hybrid" / "eval_results.json"

TRACE_SYSTEM_PROMPT = (
    "Answer the question using the retrieved context and knowledge graph. "
    "First list the supporting evidence from the graph or retrieved context, "
    "then write 'Final answer:' followed by the answer entity or entities separated by |."
)

EVAL_EXAMPLE_MODELS = {
    "qwen2.5-eval": "Qwen2.5-3B Instruct base",
    "qwen2.5-0.5b-graphrag-hybrid": "Qwen2.5-0.5B GraphRAG Hybrid",
    "qwen2.5-1.5b-graphrag-hybrid": "Qwen2.5-1.5B GraphRAG Hybrid",
    "qwen2.5-3b-graphrag-hybrid": "Qwen2.5-3B GraphRAG Hybrid",
}

REPORTED_MODEL_ROWS = [
    {
        "Model": "DistilBERT baseline",
        "Family": "baseline",
        "Params": "66M",
        "Retrieval": "span QA",
        "Training": "baseline",
        "1-hop EM": 0.449,
        "2-hop EM": 0.649,
        "3-hop EM": 0.059,
        "Overall EM": None,
        "Trace-capable": False,
    },
    {
        "Model": "Qwen2.5-1.5B RAG only",
        "Family": "student",
        "Params": "1.5B",
        "Retrieval": "RAG",
        "Training": "gold SFT",
        "1-hop EM": 0.720,
        "2-hop EM": 0.025,
        "3-hop EM": 0.020,
        "Overall EM": 0.255,
        "Trace-capable": False,
    },
    {
        "Model": "Qwen2.5-0.5B GraphRAG Gold",
        "Family": "student",
        "Params": "0.5B",
        "Retrieval": "GraphRAG",
        "Training": "gold SFT",
        "1-hop EM": 0.756,
        "2-hop EM": 0.478,
        "3-hop EM": 0.056,
        "Overall EM": 0.430,
        "Trace-capable": False,
    },
    {
        "Model": "Qwen2.5-1.5B GraphRAG Gold",
        "Family": "student",
        "Params": "1.5B",
        "Retrieval": "GraphRAG",
        "Training": "gold SFT",
        "1-hop EM": 0.778,
        "2-hop EM": 0.544,
        "3-hop EM": 0.054,
        "Overall EM": 0.459,
        "Trace-capable": False,
    },
    {
        "Model": "Qwen2.5-3B GraphRAG Gold",
        "Family": "student",
        "Params": "3B",
        "Retrieval": "GraphRAG",
        "Training": "gold SFT",
        "1-hop EM": 0.832,
        "2-hop EM": 0.586,
        "3-hop EM": 0.050,
        "Overall EM": 0.489,
        "Trace-capable": False,
    },
    {
        "Model": "Qwen2.5-0.5B GraphRAG Hybrid",
        "Family": "student",
        "Params": "0.5B",
        "Retrieval": "GraphRAG",
        "Training": "teacher evidence + gold",
        "1-hop EM": 0.572,
        "2-hop EM": 0.092,
        "3-hop EM": 0.004,
        "Overall EM": 0.223,
        "Trace-capable": True,
    },
    {
        "Model": "Qwen2.5-1.5B GraphRAG Hybrid",
        "Family": "student",
        "Params": "1.5B",
        "Retrieval": "GraphRAG",
        "Training": "teacher evidence + gold",
        "1-hop EM": 0.614,
        "2-hop EM": 0.112,
        "3-hop EM": 0.032,
        "Overall EM": 0.253,
        "Trace-capable": True,
    },
    {
        "Model": "Qwen2.5-3B GraphRAG Hybrid",
        "Family": "student",
        "Params": "3B",
        "Retrieval": "GraphRAG",
        "Training": "teacher evidence + gold",
        "1-hop EM": 0.830,
        "2-hop EM": 0.474,
        "3-hop EM": 0.070,
        "Overall EM": 0.458,
        "Trace-capable": True,
    },
]

EXAMPLE_ARTIFACTS = [
    {
        "Model": "Qwen2.5-3B Base GraphRAG",
        "Path": ROOT / "results" / "qwen2.5-eval" / "eval_examples.json",
        "Kind": "base",
        "Params": "3B",
    },
    {
        "Model": "Qwen2.5-0.5B GraphRAG Hybrid",
        "Path": ROOT / "results" / "qwen2.5-0.5b-graphrag-hybrid" / "eval_examples.json",
        "Kind": "hybrid",
        "Params": "0.5B",
    },
    {
        "Model": "Qwen2.5-1.5B GraphRAG Hybrid",
        "Path": ROOT / "results" / "qwen2.5-1.5b-graphrag-hybrid" / "eval_examples.json",
        "Kind": "hybrid",
        "Params": "1.5B",
    },
    {
        "Model": "Qwen2.5-3B GraphRAG Hybrid",
        "Path": ROOT / "results" / "qwen2.5-3b-graphrag-hybrid" / "eval_examples.json",
        "Kind": "hybrid",
        "Params": "3B",
    },
]

CHECKPOINT_CANDIDATES = [
    (
        "Qwen2.5-0.5B GraphRAG Gold",
        "qwen2.5-0.5b-graphrag-gold",
        "Smallest GraphRAG Gold student.",
    ),
    (
        "Qwen2.5-0.5B GraphRAG Hybrid",
        "qwen2.5-0.5b-graphrag-hybrid",
        "Smallest trace-trained student.",
    ),
    (
        "Qwen2.5-1.5B GraphRAG Gold",
        "qwen2.5-1.5b-graphrag-gold",
        "Mid-size GraphRAG Gold student.",
    ),
    (
        "Qwen2.5-1.5B GraphRAG Hybrid",
        "qwen2.5-1.5b-graphrag-hybrid",
        "Mid-size trace-trained student.",
    ),
    (
        "Qwen2.5-3B GraphRAG Gold",
        "qwen2.5-3b-graphrag-gold",
        "Best overall answer model in the report.",
    ),
    (
        "Qwen2.5-3B GraphRAG Hybrid",
        "qwen2.5-3b-graphrag-hybrid",
        "Best fit for live evidence-trace prompting.",
    ),
    (
        "Qwen2.5-1.5B RAG Only Gold",
        "qwen2.5-1.5b-rag-gold",
        "RAG-only baseline student for retrieval comparisons.",
    ),
]


def resolve_checkpoint_path(name: str) -> Path:
    candidates = [
        ROOT / "checkpoints" / name,
        ROOT / "checkpoints" / "checkpoints" / name,
    ]
    return next((path for path in candidates if path.exists()), candidates[0])


def get_offload_dir(name: str) -> Path:
    offload_dir = ROOT / ".offload" / name
    offload_dir.mkdir(parents=True, exist_ok=True)
    return offload_dir


st.set_page_config(
    page_title="GraphRAG Failure Mode Demo",
    page_icon=":mag:",
    layout="wide",
)


st.markdown(
    """
<style>
@import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;500;600&family=IBM+Plex+Sans:wght@400;500;600;700&display=swap');

html, body, [class*="css"] {
    font-family: 'IBM Plex Sans', sans-serif;
}

.stApp {
    background:
        radial-gradient(circle at 8% 5%, rgba(64, 180, 150, 0.14), transparent 26rem),
        radial-gradient(circle at 88% 4%, rgba(245, 177, 92, 0.12), transparent 28rem),
        linear-gradient(135deg, #0b1114 0%, #111617 45%, #14110e 100%);
}

.hero {
    border: 1px solid rgba(255,255,255,0.08);
    background: rgba(12,18,20,0.82);
    border-radius: 22px;
    padding: 1.35rem 1.5rem;
    margin-bottom: 1rem;
    box-shadow: 0 24px 80px rgba(0,0,0,0.28);
}

.hero-title {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 2.05rem;
    font-weight: 600;
    letter-spacing: -0.04em;
}

.hero-subtitle {
    max-width: 950px;
    color: #b7c7c2;
    font-size: 1.02rem;
    line-height: 1.55;
    margin-top: 0.45rem;
}

.pill {
    display: inline-block;
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.72rem;
    border: 1px solid rgba(255,255,255,0.12);
    background: rgba(255,255,255,0.06);
    border-radius: 999px;
    padding: 0.22rem 0.62rem;
    color: #d9e6df;
    margin-right: 0.35rem;
    margin-top: 0.75rem;
}

.diagnosis {
    border-left: 4px solid #6fe3bd;
    background: rgba(111, 227, 189, 0.10);
    border-radius: 14px;
    padding: 1rem 1.1rem;
    margin: 0.75rem 0 1rem 0;
}

.diagnosis-title {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.82rem;
    color: #6fe3bd;
    text-transform: uppercase;
    letter-spacing: 0.08em;
}

.diagnosis-body {
    color: #e8f1ed;
    font-size: 1.03rem;
    line-height: 1.5;
    margin-top: 0.35rem;
}

.trace-line {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.82rem;
    padding: 0.45rem 0.55rem;
    border-radius: 8px;
    margin-bottom: 0.35rem;
    background: rgba(255,255,255,0.05);
}

.supported {
    border-left: 3px solid #6fe3bd;
}

.unsupported {
    border-left: 3px solid #ff9a7a;
}

.muted {
    color: #91a09b;
}

code {
    color: #f7d38b !important;
}
</style>
""",
    unsafe_allow_html=True,
)


def load_json(path: Path):
    if not path.exists():
        return None
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


@st.cache_data(show_spinner=False)
def load_trace_items() -> list[dict]:
    data = load_json(TRACE_PATH)
    return data or []


@st.cache_data(show_spinner=False)
def load_eval_examples() -> list[dict]:
    rows = []
    for path in sorted((ROOT / "results").glob("*/eval_examples.json")):
        run_name = path.parent.name
        model_name = EVAL_EXAMPLE_MODELS.get(run_name, run_name)
        examples, load_status = load_examples(path)
        for example in examples:
            rows.append({
                **example,
                "model": model_name,
                "model_label": short_model_name(model_name),
                "run_name": run_name,
                "load_status": load_status,
            })
    return rows


@st.cache_data(show_spinner=False)
def load_local_results() -> dict:
    return load_json(LOCAL_3B_HYBRID_RESULTS) or {}


@st.cache_data(show_spinner=False)
def load_example_artifacts() -> dict[str, list[dict]]:
    loaded = {}
    for artifact in EXAMPLE_ARTIFACTS:
        examples, _ = load_examples(artifact["Path"])
        loaded[artifact["Model"]] = examples
    return loaded


def available_checkpoints() -> list[tuple[str, Path, str]]:
    rows = []
    for label, checkpoint_name, note in CHECKPOINT_CANDIDATES:
        path = resolve_checkpoint_path(checkpoint_name)
        if path.exists():
            rows.append((label, path, note))
    return rows


@st.cache_resource(show_spinner=False)
def load_live_checkpoint(model_path: str):
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    offload_dir = get_offload_dir(Path(model_path).name)
    try:
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch.float16,
            device_map="auto",
            offload_folder=str(offload_dir),
            offload_state_dict=True,
            low_cpu_mem_usage=True,
            trust_remote_code=True,
        )
    except ValueError:
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch.float32,
            device_map="cpu",
            low_cpu_mem_usage=True,
            trust_remote_code=True,
        )
    model.eval()
    return model, tokenizer


def fmt_pct(value: float | None) -> str:
    if value is None:
        return "n/a"
    return f"{value * 100:.1f}%"


def context_from_item(item: dict) -> tuple[list[str], dict]:
    sections = parse_context_sections(item.get("input", ""))
    graph_lines = sections["graph_lines"]
    retrieved_lines = sections["retrieved_lines"]
    return [*graph_lines, *retrieved_lines], sections


def parse_predicted_answers(raw_output: str) -> list[str]:
    match = re.search(r"(?is)final answer:\s*(.+)$", raw_output or "")
    answer_text = match.group(1).strip() if match else (raw_output or "").strip()
    answer_text = re.sub(r"(?i)^final answer:\s*", "", answer_text).strip()
    return [part.strip() for part in answer_text.split("|") if part.strip()]


@torch.no_grad()
def run_live_checkpoint(model, tokenizer, input_text: str, trace_output: bool = True) -> tuple[str, float]:
    messages = [
        {
            "role": "system",
            "content": TRACE_SYSTEM_PROMPT if trace_output else (
                "Answer the question using the retrieved context and knowledge graph. "
                "Return only the answer entity or entities separated by |."
            ),
        },
        {"role": "user", "content": input_text},
    ]
    prompt = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

    t0 = time.time()
    outputs = model.generate(
        **inputs,
        max_new_tokens=192 if trace_output else 64,
        do_sample=False,
        pad_token_id=tokenizer.eos_token_id,
    )
    latency_ms = (time.time() - t0) * 1000

    new_tokens = outputs[0][inputs["input_ids"].shape[1]:]
    raw_output = tokenizer.decode(new_tokens, skip_special_tokens=True).strip()
    return raw_output, latency_ms


def line_contains_any_answer(line: str, answers: list[str]) -> bool:
    line_norm = normalize_text(line)
    return any(normalize_answer(answer) in line_norm for answer in answers)


def bracketed_topic(question: str) -> str:
    match = re.search(r"\[([^\]]+)\]", question or "")
    return match.group(1) if match else "Topic entity"


def tail_entity_from_line(line: str, gold: list[str]) -> str:
    parts = re.split(r"\s*(?:->|\u2192|â†’)\s*", line)
    if len(parts) >= 3:
        candidate = parts[-1].strip()
    else:
        tokens = line.split()
        candidate = " ".join(tokens[-3:]).strip() if tokens else "Wrong Entity"

    if normalize_answer(candidate) in {normalize_answer(answer) for answer in gold}:
        return "Wrong Entity"
    return candidate or "Wrong Entity"


def apply_trace_scenario(item: dict, scenario: str) -> dict:
    metadata = item.get("metadata", {})
    question = metadata.get("question") or parse_context_sections(item.get("input", ""))["question"]
    gold = metadata.get("gold_answers", [])
    raw_output = item.get("output", "")
    context_lines, sections = context_from_item(item)
    note = "This is the recorded teacher evidence trace from the hybrid instruction data."

    if scenario == "Retrieval miss":
        context_lines = [
            line for line in context_lines
            if not line_contains_any_answer(line, gold)
        ]
        raw_output = (
            "Supporting evidence:\n"
            "- No retrieved graph triple or chunk contains the gold answer.\n\n"
            "Final answer: Unknown"
        )
        note = "What-if: gold-supporting lines are removed to show how the diagnosis becomes retrieval failure."

    elif scenario == "Unsupported evidence":
        topic = bracketed_topic(question)
        fake_answer = "Unsupported Entity"
        raw_output = (
            "Supporting evidence:\n"
            f"- {topic} -> directed_by -> {fake_answer}\n\n"
            f"Final answer: {fake_answer}"
        )
        note = "What-if: the model cites a triple that is not present in the retrieved context."

    elif scenario == "Wrong supported path":
        wrong_line = next(
            (
                line for line in context_lines
                if not line_contains_any_answer(line, gold)
            ),
            context_lines[0] if context_lines else "Topic entity -> related_to -> Wrong Entity",
        )
        wrong_answer = tail_entity_from_line(wrong_line, gold)
        raw_output = (
            "Supporting evidence:\n"
            f"- {wrong_line}\n\n"
            f"Final answer: {wrong_answer}"
        )
        note = "What-if: the evidence is real, but it supports the wrong relation/path for this question."

    elif scenario == "Answer set mismatch":
        if len(gold) > 1:
            predicted = gold[:1]
        else:
            predicted = [*gold, "Extra Entity"]
        raw_output = "Final answer: " + " | ".join(predicted)
        note = "What-if: retrieval may be fine, but exact match fails because the answer set is incomplete or too large."

    analysis = analyze_case(
        gold_answers=gold,
        raw_output=raw_output,
        context_lines=context_lines,
    )

    return {
        "question": question,
        "gold": gold,
        "raw_output": raw_output,
        "context_lines": context_lines,
        "sections": sections,
        "analysis": analysis,
        "note": note,
        "hop": metadata.get("hop", "n/a"),
    }


def answer_set_label(metrics) -> tuple[str, str]:
    if metrics.exact_match == 1.0:
        return "correct answer set", "The normalized predicted set exactly matches gold."
    if metrics.f1 > 0:
        if metrics.missing and metrics.extra:
            return (
                "partial overlap: missing + extra",
                "The model found some gold answers, missed others, and added wrong extras.",
            )
        if metrics.missing:
            return (
                "partial overlap: incomplete answer set",
                "F1 is decent but EM is zero because at least one gold answer is missing.",
            )
        if metrics.extra:
            return (
                "partial overlap: overgeneration",
                "All gold answers are present, but extra wrong answers break exact match.",
            )
    return "exact miss", "The prediction has no normalized overlap with the gold answer set."


def render_diagnosis(analysis: dict) -> None:
    st.markdown(
        f"""
<div class="diagnosis">
  <div class="diagnosis-title">{analysis["failure_mode"]}</div>
  <div class="diagnosis-body">{analysis["explanation"]}</div>
</div>
""",
        unsafe_allow_html=True,
    )


def answer_set_category(metrics) -> str:
    if metrics.exact_match == 1.0:
        return "correct"
    if metrics.f1 == 0:
        return "exact miss"
    if metrics.missing and metrics.extra:
        return "missing + extra"
    if metrics.missing:
        return "incomplete answer set"
    if metrics.extra:
        return "overgeneration"
    return "format / set mismatch"


def summarize_examples_by_model(example_artifacts: dict[str, list[dict]]) -> pd.DataFrame:
    rows = []
    for artifact in EXAMPLE_ARTIFACTS:
        model = artifact["Model"]
        examples = example_artifacts.get(model, [])
        for example in examples:
            metrics = answer_metrics(example.get("pred", []), example.get("gold", []))
            rows.append(
                {
                    "Model": model,
                    "Params": artifact["Params"],
                    "Kind": artifact["Kind"],
                    "Failure category": answer_set_category(metrics),
                    "EM": metrics.exact_match,
                    "F1": metrics.f1,
                    "Gold answers": len(example.get("gold", [])),
                    "Predicted answers": len(example.get("pred", [])),
                    "Missing": len(metrics.missing),
                    "Extra": len(metrics.extra),
                }
            )
    return pd.DataFrame(rows)


def render_altair_chart(chart, fallback_df: pd.DataFrame | None = None) -> None:
    try:
        st.altair_chart(chart, use_container_width=True)
    except Exception:
        if fallback_df is not None:
            st.bar_chart(fallback_df)
        else:
            st.warning("Chart rendering failed, but the table below still contains the values.")


def render_all_model_comparison_tab(example_artifacts: dict[str, list[dict]]) -> None:
    st.markdown(
        "This view compares the full reported model set. The bars are aggregate EM, while the lower section uses "
        "saved per-example predictions where available for answer-set error analysis."
    )

    reported_df = pd.DataFrame(REPORTED_MODEL_ROWS)
    display_df = reported_df.copy()
    display_df["Trace-capable"] = display_df["Trace-capable"].map({True: "yes", False: "no"})

    st.dataframe(
        display_df,
        hide_index=True,
        use_container_width=True,
        column_config={
            "1-hop EM": st.column_config.ProgressColumn("1-hop EM", min_value=0, max_value=1, format="%.3f"),
            "2-hop EM": st.column_config.ProgressColumn("2-hop EM", min_value=0, max_value=1, format="%.3f"),
            "3-hop EM": st.column_config.ProgressColumn("3-hop EM", min_value=0, max_value=1, format="%.3f"),
            "Overall EM": st.column_config.ProgressColumn("Overall EM", min_value=0, max_value=1, format="%.3f"),
        },
    )

    try:
        import altair as alt

        long_df = reported_df.melt(
            id_vars=["Model", "Retrieval", "Training", "Params"],
            value_vars=["1-hop EM", "2-hop EM", "3-hop EM"],
            var_name="Hop",
            value_name="EM",
        )
        chart = (
            alt.Chart(long_df)
            .mark_bar(cornerRadiusTopLeft=4, cornerRadiusTopRight=4)
            .encode(
                x=alt.X("Model:N", sort=None, axis=alt.Axis(labelAngle=-35)),
                y=alt.Y("EM:Q", scale=alt.Scale(domain=[0, 1])),
                color=alt.Color("Hop:N", scale=alt.Scale(range=["#6fe3bd", "#f4c06a", "#ff8f72"])),
                tooltip=["Model", "Params", "Retrieval", "Training", "Hop", alt.Tooltip("EM:Q", format=".3f")],
            )
            .properties(height=360)
        )
        render_altair_chart(chart)

        overall_df = reported_df.dropna(subset=["Overall EM"]).sort_values("Overall EM", ascending=False)
        overall_chart = (
            alt.Chart(overall_df)
            .mark_bar(cornerRadiusTopRight=5, cornerRadiusBottomRight=5)
            .encode(
                y=alt.Y("Model:N", sort="-x"),
                x=alt.X("Overall EM:Q", scale=alt.Scale(domain=[0, 0.55])),
                color=alt.Color("Training:N", scale=alt.Scale(range=["#6fe3bd", "#f4c06a", "#ff8f72"])),
                tooltip=["Model", "Params", "Retrieval", "Training", alt.Tooltip("Overall EM:Q", format=".3f")],
            )
            .properties(height=300)
        )
        render_altair_chart(overall_chart)
    except Exception:
        st.bar_chart(reported_df.set_index("Model")[["1-hop EM", "2-hop EM", "3-hop EM"]])

    st.markdown("**Immediate error-analysis reads from the aggregate comparison**")
    st.write(
        "RAG-only collapses on 2-hop, which is strong evidence of retrieval/path-coverage weakness rather than just generation weakness."
    )
    st.write(
        "GraphRAG Gold scales best overall with 3B, while Hybrid only wins 3-hop EM; that suggests evidence supervision may help deep traces but can inject supervision noise elsewhere."
    )
    st.write(
        "The large 3-hop gap between EM and expected answer burden should be treated as multi-answer/output-burden pressure, not simply deeper reasoning failure."
    )

    example_df = summarize_examples_by_model(example_artifacts)
    if example_df.empty:
        st.info("No saved per-example prediction files were found for model-level failure breakdowns.")
        return

    st.markdown("### Saved Prediction Failure Breakdown")
    st.caption(
        "This section is based only on saved `eval_examples.json` artifacts. It diagnoses answer-set behavior, "
        "not retrieval coverage, because those files do not include the retrieved graph/context."
    )

    category_counts = (
        example_df.groupby(["Model", "Failure category"])
        .size()
        .reset_index(name="Examples")
    )
    totals = category_counts.groupby("Model")["Examples"].transform("sum")
    category_counts["Share"] = category_counts["Examples"] / totals

    try:
        import altair as alt

        stacked = (
            alt.Chart(category_counts)
            .mark_bar()
            .encode(
                x=alt.X("Share:Q", axis=alt.Axis(format="%")),
                y=alt.Y("Model:N", sort=None),
                color=alt.Color(
                    "Failure category:N",
                    scale=alt.Scale(
                        domain=[
                            "correct",
                            "incomplete answer set",
                            "overgeneration",
                            "missing + extra",
                            "format / set mismatch",
                            "exact miss",
                        ],
                        range=["#6fe3bd", "#f4c06a", "#ffb36b", "#ff8f72", "#d0d8d4", "#7a8791"],
                    ),
                ),
                tooltip=["Model", "Failure category", "Examples", alt.Tooltip("Share:Q", format=".1%")],
            )
            .properties(height=240)
        )
        render_altair_chart(stacked)

        burden_df = (
            example_df.groupby("Model")
            .agg(
                avg_gold_answers=("Gold answers", "mean"),
                avg_missing=("Missing", "mean"),
                avg_extra=("Extra", "mean"),
                avg_f1=("F1", "mean"),
            )
            .reset_index()
        )
        burden_long = burden_df.melt(
            id_vars=["Model", "avg_f1"],
            value_vars=["avg_gold_answers", "avg_missing", "avg_extra"],
            var_name="Signal",
            value_name="Average count",
        )
        burden_chart = (
            alt.Chart(burden_long)
            .mark_bar(cornerRadiusTopLeft=4, cornerRadiusTopRight=4)
            .encode(
                x=alt.X("Model:N", sort=None, axis=alt.Axis(labelAngle=-25)),
                y="Average count:Q",
                color=alt.Color("Signal:N", scale=alt.Scale(range=["#6fe3bd", "#f4c06a", "#ff8f72"])),
                tooltip=["Model", "Signal", alt.Tooltip("Average count:Q", format=".2f")],
            )
            .properties(height=280)
        )
        render_altair_chart(burden_chart)
    except Exception:
        pivot = category_counts.pivot_table(
            index="Model",
            columns="Failure category",
            values="Share",
            fill_value=0,
        )
        st.bar_chart(pivot)

    summary = (
        example_df.groupby("Model")
        .agg(
            saved_examples=("Model", "size"),
            sample_em=("EM", "mean"),
            sample_f1=("F1", "mean"),
            avg_gold_answers=("Gold answers", "mean"),
            avg_missing=("Missing", "mean"),
            avg_extra=("Extra", "mean"),
        )
        .reset_index()
    )
    st.dataframe(
        summary,
        hide_index=True,
        use_container_width=True,
        column_config={
            "sample_em": st.column_config.ProgressColumn("sample EM", min_value=0, max_value=1, format="%.3f"),
            "sample_f1": st.column_config.ProgressColumn("sample F1", min_value=0, max_value=1, format="%.3f"),
        },
    )


def render_answer_columns(question: str, gold: list[str], analysis: dict) -> None:
    left, middle, right = st.columns([1.5, 1, 1])
    with left:
        st.markdown("**Question**")
        st.write(question)
    with middle:
        st.markdown("**Gold answer set**")
        st.code(" | ".join(gold) if gold else "(none)", language="text")
    with right:
        st.markdown("**Predicted answer set**")
        predicted = analysis["predicted"]
        st.code(" | ".join(predicted) if predicted else "(empty)", language="text")


def render_metric_strip(analysis: dict) -> None:
    metrics = analysis["metrics"]
    support_ratio = analysis["evidence_support"]["support_ratio"]
    coverage = analysis["coverage"]["gold"]

    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Exact match", f"{metrics['em']:.0f}")
    c2.metric("F1", fmt_pct(metrics["f1"]))
    c3.metric("Gold in context", "all" if coverage["all"] else "partial" if coverage["any"] else "none")
    c4.metric("Evidence support", fmt_pct(support_ratio))
    c5.metric("Evidence lines", str(analysis["evidence_support"]["total"]))


def render_teacher_trace_metrics(analysis: dict, gold: list[str]) -> None:
    evidence_coverage = answer_coverage_in_lines(gold, analysis["evidence"])
    grounded_coverage = answer_coverage_in_lines(
        gold, analysis["evidence_support"]["supported"]
    )
    compression_gap = len(gold) - grounded_coverage["covered_count"]

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Trace covers gold", fmt_pct(evidence_coverage["coverage_rate"]))
    c2.metric("Grounded trace covers gold", fmt_pct(grounded_coverage["coverage_rate"]))
    c3.metric("Grounded compression gap", str(compression_gap))
    c4.metric(
        "Direct-answer artifact",
        "yes" if analysis["metrics"]["em"] == 1.0 and not grounded_coverage["any"] else "no",
    )


def render_evidence(analysis: dict) -> None:
    evidence = analysis["evidence"]
    support = analysis["evidence_support"]
    if not evidence:
        st.info("No explicit evidence trace was emitted for this case.")
        return

    supported = set(support["supported"])
    for line in evidence:
        status_class = "supported" if line in supported else "unsupported"
        status = "supported" if line in supported else "unsupported"
        st.markdown(
            f'<div class="trace-line {status_class}"><span class="muted">{status}:</span> {line}</div>',
            unsafe_allow_html=True,
        )


def render_context(case: dict) -> None:
    sections = case["sections"]
    left, right = st.columns(2)
    with left:
        st.markdown("**Knowledge graph context**")
        for line in sections["graph_lines"][:18]:
            st.code(line, language="text")
        if len(sections["graph_lines"]) > 18:
            st.caption(f"{len(sections['graph_lines']) - 18} more graph lines hidden.")
    with right:
        st.markdown("**FAISS retrieved context**")
        for line in sections["retrieved_lines"][:10]:
            st.code(line, language="text")
        if len(sections["retrieved_lines"]) > 10:
            st.caption(f"{len(sections['retrieved_lines']) - 10} more retrieved lines hidden.")


def render_model_status() -> None:
    local_results = load_local_results()
    checkpoints = available_checkpoints()

    st.markdown(
        """
<div class="hero">
  <div class="hero-title">GraphRAG Failure Mode Analysis MVP</div>
  <div class="hero-subtitle">
    A performance snapshot says what happened. This demo shows why it likely happened by checking retrieval coverage,
    evidence validity, path faithfulness, and answer-set behavior.
  </div>
  <span class="pill">best student family: Qwen2.5-3B</span>
  <span class="pill">analysis type: error source / failure mode</span>
  <span class="pill">mode: replayable MVP</span>
</div>
""",
        unsafe_allow_html=True,
    )

    if checkpoints:
        names = ", ".join(name for name, _, _ in checkpoints)
        st.success(f"Local checkpoint detected: {names}. Live parent-vs-student comparison is enabled in this app.")
    else:
        st.warning(
            "No local `checkpoints/`, `data/raw/`, or `data/faiss/` artifacts are committed here, "
            "so this MVP runs in replay mode from bundled result and instruction artifacts."
        )

    rows = [
        {
            "Model": "Qwen2.5-3B GraphRAG Gold",
            "Use in demo story": "Best overall answer model from the report",
            "Overall EM": "0.489",
            "3-hop EM": "0.050",
        },
        {
            "Model": "Qwen2.5-3B GraphRAG Hybrid",
            "Use in demo story": "Best local evidence-trace/result artifact",
            "Overall EM": f"{local_results.get('overall/EM', 0.458):.3f}",
            "3-hop EM": f"{local_results.get('test_3hop/EM', 0.070):.3f}",
        },
    ]
    st.dataframe(rows, hide_index=True, use_container_width=True)


def render_trace_tab(trace_items: list[dict]) -> None:
    if not trace_items:
        st.error(f"Trace artifact not found: {TRACE_PATH}")
        return

    st.markdown(
        "This panel demonstrates the full failure-mode logic because each case includes context, evidence, and gold answers."
    )

    default_index = next(
        (
            i for i, item in enumerate(trace_items)
            if len(item.get("metadata", {}).get("gold_answers", [])) > 1
        ),
        0,
    )
    labels = [
        f"{item.get('metadata', {}).get('hop', '?')}-hop | {item.get('metadata', {}).get('question', 'unknown question')}"
        for item in trace_items[:300]
    ]

    col_a, col_b = st.columns([2, 1])
    with col_a:
        selected_label = st.selectbox("Demo question", labels, index=default_index)
    with col_b:
        scenario = st.selectbox(
            "Scenario",
            [
                "Recorded trace",
                "Retrieval miss",
                "Unsupported evidence",
                "Wrong supported path",
                "Answer set mismatch",
            ],
        )

    selected_item = trace_items[labels.index(selected_label)]
    case = apply_trace_scenario(selected_item, scenario)
    analysis = case["analysis"]

    st.caption(case["note"])
    render_answer_columns(case["question"], case["gold"], analysis)
    render_metric_strip(analysis)
    render_teacher_trace_metrics(analysis, case["gold"])
    render_diagnosis(analysis)

    left, right = st.columns([1, 1])
    with left:
        st.markdown("**Evidence trace**")
        render_evidence(analysis)
    with right:
        st.markdown("**Answer-set deltas**")
        missing = analysis["metrics"]["missing"]
        extra = analysis["metrics"]["extra"]
        st.write(f"Missing gold answers: `{', '.join(missing) if missing else 'none'}`")
        st.write(f"Extra predicted answers: `{', '.join(extra) if extra else 'none'}`")
        st.write(f"Gold answers covered by context: `{len(analysis['coverage']['gold']['covered'])}`")

    with st.expander("Show graph and retrieval context", expanded=False):
        render_context(case)


def render_actual_error_tab(eval_examples: list[dict]) -> None:
    if not eval_examples:
        st.error("No `results/*/eval_examples.json` artifacts were found.")
        return

    errors = [example for example in eval_examples if float(example.get("em", 0)) == 0.0]
    errors.sort(key=lambda item: (float(item.get("f1", 0)), len(item.get("gold", []))), reverse=True)

    if not errors:
        st.success("No errors found in the saved examples.")
        return

    st.markdown(
        "This panel compares saved prediction errors across available model example files. Most committed examples do not include retrieved context, "
        "so the diagnosis here is primarily answer-set behavior unless future evals are run with `--save_context`."
    )

    model_options = sorted({example["model_label"] for example in errors})
    selected_model = st.selectbox("Model", model_options, index=model_options.index("Hybrid 3B") if "Hybrid 3B" in model_options else 0)
    model_errors = [example for example in errors if example["model_label"] == selected_model]

    labels = [
        f"F1={example.get('f1', 0):.2f} | gold={len(example.get('gold', []))} pred={len(example.get('pred', []))} | {example.get('question')}"
        for example in model_errors
    ]
    selected_label = st.selectbox("Actual saved model error", labels)
    example = model_errors[labels.index(selected_label)]

    metrics = answer_metrics(example.get("pred", []), example.get("gold", []))
    label, explanation = answer_set_label(metrics)

    context_lines = []
    if example.get("input_text"):
        sections = parse_context_sections(example.get("input_text", ""))
        context_lines.extend(sections["graph_lines"])
        context_lines.extend(sections["retrieved_lines"])
    context_lines.extend(str(chunk) for chunk in example.get("retrieved_chunks", []))
    if context_lines:
        contextual = analyze_case(
            gold_answers=example.get("gold", []),
            raw_output=example.get("raw_output", ""),
            predicted_answers=example.get("pred", []),
            context_lines=context_lines,
        )
        label = contextual["failure_mode"]
        explanation = contextual["explanation"]

    analysis = {
        "predicted": example.get("pred", []),
        "metrics": {
            "em": metrics.exact_match,
            "f1": metrics.f1,
            "missing": metrics.missing,
            "extra": metrics.extra,
        },
    }

    render_answer_columns(example.get("question", ""), example.get("gold", []), analysis)

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Exact match", f"{example.get('em', 0):.0f}")
    c2.metric("F1", fmt_pct(float(example.get("f1", 0))))
    c3.metric("Gold answers", str(len(example.get("gold", []))))
    c4.metric("Predicted answers", str(len(example.get("pred", []))))
    st.caption(f"Source: `{example.get('run_name')}` / `{example.get('load_status')}`")

    st.markdown(
        f"""
<div class="diagnosis">
  <div class="diagnosis-title">{label}</div>
  <div class="diagnosis-body">{explanation}</div>
</div>
""",
        unsafe_allow_html=True,
    )

    left, right = st.columns(2)
    with left:
        st.markdown("**Missing gold answers**")
        if metrics.missing:
            for answer in metrics.missing:
                st.code(answer, language="text")
        else:
            st.caption("None")
    with right:
        st.markdown("**Extra predicted answers**")
        if metrics.extra:
            for answer in metrics.extra:
                st.code(answer, language="text")
        else:
            st.caption("None")

    with st.expander("Raw model output", expanded=False):
        st.code(example.get("raw_output", ""), language="text")
        st.caption(f"Latency: {example.get('latency_ms', 'n/a')} ms")


def reported_metrics_for_label(model_label: str) -> dict | None:
    for row in REPORTED_MODEL_ROWS:
        if short_model_name(row["Model"]) == model_label or row["Model"] == model_label:
            return row
    return None


def render_prediction_sample(example: dict, index: int) -> None:
    pred = example.get("pred") or []
    gold = example.get("gold") or []
    metrics = answer_metrics(pred, gold)
    label, explanation = answer_set_label(metrics)

    with st.container(border=True):
        st.markdown(f"**Example {index}: {example.get('question', 'unknown question')}**")
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("EM", f"{metrics.exact_match:.0f}")
        c2.metric("F1", fmt_pct(metrics.f1))
        c3.metric("Gold count", str(len(gold)))
        c4.metric("Pred count", str(len(pred)))

        left, right = st.columns(2)
        with left:
            st.markdown("Gold set")
            st.code(" | ".join(gold) if gold else "(none)", language="text")
        with right:
            st.markdown("LLM response")
            st.code(" | ".join(pred) if pred else example.get("raw_output", "(empty)"), language="text")

        st.caption(f"{label}: {explanation}")
        if metrics.missing:
            st.write(f"Missing: `{', '.join(metrics.missing)}`")
        if metrics.extra:
            st.write(f"Extra: `{', '.join(metrics.extra)}`")


def render_trace_sample(item: dict, index: int) -> None:
    case = apply_trace_scenario(item, "Recorded trace")
    analysis = case["analysis"]
    with st.container(border=True):
        st.markdown(f"**Trace example {index}: {case['question']}**")
        render_answer_columns(case["question"], case["gold"], analysis)
        render_metric_strip(analysis)
        render_teacher_trace_metrics(analysis, case["gold"])
        render_evidence(analysis)


def render_parent_student_tab(trace_items: list[dict]) -> None:
    st.markdown(
        "Pick a teacher-trace example by hop, then run a local student checkpoint on the exact same context. "
        "This is the cleanest app-style comparison available in the repo because the teacher trace and gold answers are already stored."
    )

    checkpoints = available_checkpoints()
    if not checkpoints:
        st.warning("No local checkpoints were found, so live student inference is unavailable.")
        return
    if not trace_items:
        st.warning(f"Teacher trace artifact not found: {TRACE_PATH}")
        return

    hop = st.selectbox("Hop", [1, 2, 3], key="parent_student_hop")
    hop_items = [item for item in trace_items if item.get("metadata", {}).get("hop") == hop]
    if not hop_items:
        st.info(f"No teacher-trace items found for {hop}-hop.")
        return

    checkpoint_labels = [name for name, _, _ in checkpoints]
    selected_label = st.selectbox("Student checkpoint", checkpoint_labels, key="parent_student_checkpoint")
    selected_checkpoint = next(path for name, path, _ in checkpoints if name == selected_label)

    labels = [
        item.get("metadata", {}).get("question", f"Example {idx + 1}")
        for idx, item in enumerate(hop_items[:200])
    ]
    chosen_question = st.selectbox("Question", labels, key="parent_student_question")
    selected_item = hop_items[labels.index(chosen_question)]
    teacher_case = apply_trace_scenario(selected_item, "Recorded trace")

    use_trace_output = st.toggle("Ask student for evidence trace too", value=True, key="parent_student_trace_toggle")
    if st.button("Run selected student on this teacher example", type="primary", key="run_parent_student"):
        with st.spinner(f"Running {selected_label}..."):
            model, tokenizer = load_live_checkpoint(str(selected_checkpoint))
            raw_output, latency_ms = run_live_checkpoint(
                model,
                tokenizer,
                selected_item.get("input", ""),
                trace_output=use_trace_output,
            )
        st.session_state["parent_student_result"] = {
            "question": teacher_case["question"],
            "hop": hop,
            "model_label": selected_label,
            "raw_output": raw_output,
            "latency_ms": latency_ms,
            "gold": teacher_case["gold"],
            "context_lines": teacher_case["context_lines"],
            "teacher_case": teacher_case,
            "student_analysis": analyze_case(
                gold_answers=teacher_case["gold"],
                raw_output=raw_output,
                predicted_answers=parse_predicted_answers(raw_output),
                context_lines=teacher_case["context_lines"],
            ),
        }

    result = st.session_state.get("parent_student_result")
    if not result or result.get("question") != teacher_case["question"] or result.get("model_label") != selected_label:
        st.info("Select a hop/question and run a local student checkpoint to compare against the stored teacher trace.")
        return

    teacher_analysis = teacher_case["analysis"]
    student_analysis = result["student_analysis"]

    top_left, top_right = st.columns(2)
    with top_left:
        st.markdown("### Teacher / parent reference")
        render_answer_columns(teacher_case["question"], teacher_case["gold"], teacher_analysis)
        render_metric_strip(teacher_analysis)
        render_teacher_trace_metrics(teacher_analysis, teacher_case["gold"])
        render_diagnosis(teacher_analysis)
        render_evidence(teacher_analysis)
    with top_right:
        st.markdown(f"### Student checkpoint: {selected_label}")
        render_answer_columns(result["question"], result["gold"], student_analysis)
        render_metric_strip(student_analysis)
        c1, c2 = st.columns(2)
        c1.metric("Latency", f"{result['latency_ms']:.0f} ms")
        c2.metric("Hop", str(result["hop"]))
        render_diagnosis(student_analysis)
        st.markdown("**Student raw output**")
        st.code(result["raw_output"], language="text")
        if student_analysis["evidence"]:
            st.markdown("**Student evidence trace**")
            render_evidence(student_analysis)

    with st.expander("Show shared graph and retrieval context", expanded=False):
        render_context(teacher_case)


def render_hop_explorer_tab(eval_examples: list[dict], trace_items: list[dict]) -> None:
    st.markdown(
        "Goal: select a model and hop, then inspect three questions with LLM output, gold answers, sample performance, and overall performance."
    )

    hop_labeled = [
        example for example in eval_examples
        if example.get("hop") is not None and example.get("model_label")
    ]

    if hop_labeled:
        model_options = sorted({example["model_label"] for example in hop_labeled})
        selected_model = st.selectbox("Model", model_options, key="hop_explorer_model")
        model_rows = [example for example in hop_labeled if example["model_label"] == selected_model]
        hop_options = sorted({int(example["hop"]) for example in model_rows})
        selected_hop = st.selectbox("Hop", hop_options, key="hop_explorer_hop")
        subset = [example for example in model_rows if int(example["hop"]) == selected_hop]
        subset.sort(key=lambda item: (float(item.get("em", 0)), -float(item.get("f1", 0))))

        sample = subset[:3]
        sample_em = sum(float(example.get("em", 0)) for example in subset) / len(subset)
        sample_f1 = sum(float(example.get("f1", 0)) for example in subset) / len(subset)
        reported = reported_metrics_for_label(selected_model) or {}

        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Subset examples", str(len(subset)))
        c2.metric("Subset EM", fmt_pct(sample_em))
        c3.metric("Subset F1", fmt_pct(sample_f1))
        c4.metric("Reported hop EM", fmt_pct(reported.get(f"{selected_hop}-hop EM")))

        overall = reported.get("Overall EM")
        st.caption(
            f"Overall reported EM for {selected_model}: {overall:.3f}" if overall is not None else
            "Overall reported EM is unavailable for this model."
        )

        for i, example in enumerate(sample, start=1):
            render_prediction_sample(example, i)
        return

    st.warning(
        "The committed `eval_examples.json` files do not contain hop/context fields, so true per-hop model-output exploration "
        "requires re-running evaluation with `--save_context --examples_limit 0`."
    )
    st.code(
        "python run_all.py --eval_only --reeval --examples_limit 0 --save_context\n"
        "python src/evaluation/error_analysis_report.py",
        language="powershell",
    )

    st.markdown(
        "Fallback shown below: per-hop **teacher/evidence trace** samples from the instruction artifacts. "
        "These are useful for explaining trace quality and retrieval coverage, but they are not before/after model predictions."
    )
    hop = st.selectbox("Hop", [1, 2, 3], key="trace_hop_fallback")
    candidates = [item for item in trace_items if item.get("metadata", {}).get("hop") == hop]
    for i, item in enumerate(candidates[:3], start=1):
        render_trace_sample(item, i)


def main() -> None:
    render_model_status()

    trace_items = load_trace_items()
    eval_examples = load_eval_examples()
    example_artifacts = load_example_artifacts()

    compare_tab, parent_tab, trace_tab, hop_tab, error_tab = st.tabs(
        ["All-model comparison", "Teacher vs student", "Evidence trace diagnosis", "Hop explorer", "Saved model errors"]
    )

    with compare_tab:
        render_all_model_comparison_tab(example_artifacts)

    with parent_tab:
        render_parent_student_tab(trace_items)

    with trace_tab:
        render_trace_tab(trace_items)

    with hop_tab:
        render_hop_explorer_tab(eval_examples, trace_items)

    with error_tab:
        render_actual_error_tab(eval_examples)


if __name__ == "__main__":
    main()
