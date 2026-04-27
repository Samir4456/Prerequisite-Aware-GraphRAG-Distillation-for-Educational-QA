"""
app.py — Pocket GraphRAG Demo
Side-by-side comparison: DistilBERT baseline vs Qwen2.5 student

Run with:
    streamlit run app.py
"""

import sys
import time
from pathlib import Path

import pandas as pd
import streamlit as st
import torch
import numpy as np

sys.path.insert(0, "src/data")
sys.path.insert(0, "src/graph")
sys.path.insert(0, "src/retrieval")


def resolve_checkpoint_path(*parts: str) -> Path:
    """
    Support both checkpoint layouts:
    - checkpoints/<model_name>
    - checkpoints/checkpoints/<model_name>
    """
    candidates = [
        Path("checkpoints", *parts),
        Path("checkpoints", "checkpoints", *parts),
    ]
    return next((path for path in candidates if path.exists()), candidates[0])


def get_offload_dir(name: str) -> Path:
    offload_dir = Path(".offload") / name
    offload_dir.mkdir(parents=True, exist_ok=True)
    return offload_dir


def available_qwen_models():
    return [
        {
            "label": "Qwen2.5-3B GraphRAG Hybrid",
            "path": resolve_checkpoint_path("qwen2.5-3b-graphrag-hybrid"),
            "params": "3B (LoRA + traces)",
            "trace_native": True,
            "metrics_label": "Qwen2.5-3B GraphRAG Hybrid",
        },
        {
            "label": "Qwen2.5-3B GraphRAG Gold",
            "path": resolve_checkpoint_path("qwen2.5-3b-graphrag-gold"),
            "params": "3B (LoRA)",
            "trace_native": False,
            "metrics_label": "Qwen2.5-3B GraphRAG Gold",
        },
        {
            "label": "Qwen2.5-1.5B GraphRAG Hybrid",
            "path": resolve_checkpoint_path("qwen2.5-1.5b-graphrag-hybrid"),
            "params": "1.5B (LoRA + traces)",
            "trace_native": True,
            "metrics_label": "Qwen2.5-1.5B GraphRAG Hybrid",
        },
        {
            "label": "Qwen2.5-1.5B GraphRAG Gold",
            "path": resolve_checkpoint_path("qwen2.5-1.5b-graphrag-gold"),
            "params": "1.5B (LoRA)",
            "trace_native": False,
            "metrics_label": "Qwen2.5-1.5B GraphRAG Gold",
        },
        {
            "label": "Qwen2.5-0.5B GraphRAG Hybrid",
            "path": resolve_checkpoint_path("qwen2.5-0.5b-graphrag-hybrid"),
            "params": "0.5B (LoRA + traces)",
            "trace_native": True,
            "metrics_label": "Qwen2.5-0.5B GraphRAG Hybrid",
        },
        {
            "label": "Qwen2.5-0.5B GraphRAG Gold",
            "path": resolve_checkpoint_path("qwen2.5-0.5b-graphrag-gold"),
            "params": "0.5B (LoRA)",
            "trace_native": False,
            "metrics_label": "Qwen2.5-0.5B GraphRAG Gold",
        },
        {
            "label": "Qwen2.5-1.5B RAG Only Gold",
            "path": resolve_checkpoint_path("qwen2.5-1.5b-rag-gold"),
            "params": "1.5B (RAG-only)",
            "trace_native": False,
            "metrics_label": "Qwen2.5-1.5B RAG only Gold",
        },
    ]


@st.cache_data
def load_metrics_table() -> pd.DataFrame:
    metrics_path = Path("results/error_analysis/model_metrics_all.csv")
    if not metrics_path.exists():
        return pd.DataFrame()
    return pd.read_csv(metrics_path)


@st.cache_data
def load_trace_summary_table() -> pd.DataFrame:
    path = Path("results/error_analysis/dataset_trace_summary_by_hop.csv")
    if not path.exists():
        return pd.DataFrame()
    return pd.read_csv(path)


@st.cache_data
def load_answer_breakdown_table() -> pd.DataFrame:
    path = Path("results/error_analysis/answer_set_error_breakdown.csv")
    if not path.exists():
        return pd.DataFrame()
    return pd.read_csv(path)


def metrics_for_model(model_label: str) -> dict:
    df = load_metrics_table()
    if df.empty:
        return {}
    row = df[df["model"] == model_label]
    if row.empty:
        return {}
    return row.iloc[0].to_dict()


def split_trace_and_answer(raw_output: str) -> tuple[str, str]:
    if not raw_output:
        return "", ""
    match = __import__("re").search(r"(?is)(.*?)(?:final answer:\s*)(.+)$", raw_output.strip())
    if match:
        return match.group(1).strip(), match.group(2).strip()
    return "", raw_output.strip()

# ─────────────────────────────────────────────
# Page config
# ─────────────────────────────────────────────

st.set_page_config(
    page_title="Pocket GraphRAG",
    page_icon="🔍",
    layout="wide",
)

# ─────────────────────────────────────────────
# Custom CSS
# ─────────────────────────────────────────────

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;500&family=IBM+Plex+Sans:wght@300;400;500;600&display=swap');

html, body, [class*="css"] {
    font-family: 'IBM Plex Sans', sans-serif;
}

.stApp {
    background: #0a0a0f;
    color: #e8e8f0;
}

.main-title {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 2.2rem;
    font-weight: 500;
    color: #e8e8f0;
    letter-spacing: -0.02em;
    margin-bottom: 0.2rem;
}

.subtitle {
    font-size: 0.9rem;
    color: #6b6b80;
    font-family: 'IBM Plex Mono', monospace;
    letter-spacing: 0.05em;
    margin-bottom: 2rem;
}

.model-card {
    background: #12121a;
    border: 1px solid #1e1e2e;
    border-radius: 12px;
    padding: 1.5rem;
    margin-bottom: 1rem;
}

.model-header {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.75rem;
    font-weight: 500;
    letter-spacing: 0.12em;
    text-transform: uppercase;
    margin-bottom: 1rem;
    padding-bottom: 0.75rem;
    border-bottom: 1px solid #1e1e2e;
}

.distilbert-header { color: #7c85ff; }
.qwen-header { color: #4fffb0; }

.answer-box {
    background: #0d0d16;
    border-radius: 8px;
    padding: 1rem 1.25rem;
    margin: 0.75rem 0;
    border-left: 3px solid;
}

.distilbert-answer { border-color: #7c85ff; }
.qwen-answer { border-color: #4fffb0; }

.answer-text {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 1.1rem;
    font-weight: 500;
    color: #e8e8f0;
}

.metric-row {
    display: flex;
    gap: 1rem;
    margin-top: 0.75rem;
    flex-wrap: wrap;
}

.metric-pill {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.72rem;
    padding: 0.25rem 0.6rem;
    border-radius: 20px;
    background: #1a1a28;
    color: #6b6b80;
    border: 1px solid #1e1e2e;
}

.metric-pill span {
    color: #e8e8f0;
    font-weight: 500;
}

.section-label {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.68rem;
    letter-spacing: 0.1em;
    text-transform: uppercase;
    color: #3d3d52;
    margin-bottom: 0.5rem;
    margin-top: 1rem;
}

.triple-item {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.78rem;
    color: #9090a8;
    padding: 0.2rem 0;
    line-height: 1.6;
}

.triple-item .entity { color: #c8c8e0; }
.triple-item .rel { color: #5555aa; }

.chunk-item {
    font-size: 0.82rem;
    color: #7070a0;
    padding: 0.3rem 0;
    border-bottom: 1px solid #1a1a28;
    line-height: 1.5;
}

.entity-tag {
    display: inline-block;
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.75rem;
    background: #1a1a2e;
    border: 1px solid #2a2a4a;
    color: #8888cc;
    padding: 0.15rem 0.5rem;
    border-radius: 4px;
    margin-bottom: 0.5rem;
}

.hop-badge {
    display: inline-block;
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.65rem;
    background: #1a2a1a;
    border: 1px solid #2a4a2a;
    color: #4a8a4a;
    padding: 0.1rem 0.4rem;
    border-radius: 3px;
    margin-left: 0.5rem;
}

.sample-hop-label {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.66rem;
    letter-spacing: 0.08em;
    text-transform: uppercase;
    margin-bottom: 0.45rem;
}

.hop1 { color: #6fe3bd; }
.hop2 { color: #f4c06a; }
.hop3 { color: #ff8f72; }

.stTextInput > div > div > input {
    background: #12121a !important;
    border: 1px solid #1e1e2e !important;
    border-radius: 8px !important;
    color: #e8e8f0 !important;
    font-family: 'IBM Plex Sans', sans-serif !important;
    font-size: 1rem !important;
    padding: 0.75rem 1rem !important;
}

.stButton > button {
    background: #1e1e3a !important;
    border: 1px solid #3333aa !important;
    color: #9090ff !important;
    font-family: 'IBM Plex Mono', monospace !important;
    font-size: 0.8rem !important;
    letter-spacing: 0.05em !important;
    border-radius: 6px !important;
    padding: 0.5rem 1.5rem !important;
    transition: all 0.15s !important;
}

.stButton > button:hover {
    background: #252550 !important;
    border-color: #5555cc !important;
    color: #b0b0ff !important;
}

.stSelectbox > div > div {
    background: #12121a !important;
    border: 1px solid #1e1e2e !important;
    color: #e8e8f0 !important;
    border-radius: 8px !important;
}

hr { border-color: #1e1e2e !important; }

.status-ok {
    color: #4fffb0;
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.8rem;
}
.status-err {
    color: #ff6b6b;
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.8rem;
}

.divider {
    border: none;
    border-top: 1px solid #1e1e2e;
    margin: 1.5rem 0;
}
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────
# Load resources (cached)
# ─────────────────────────────────────────────

@st.cache_resource
def load_kb_and_index():
    from load_kb import load_kb
    from faiss_index import load_index
    triples, adjacency = load_kb("data/raw/kb.txt")
    index, corpus = load_index("data/faiss/index.bin", "data/faiss/corpus.pkl")
    return adjacency, index, corpus


@st.cache_resource
def load_distilbert():
    from transformers import DistilBertForQuestionAnswering, DistilBertTokenizerFast
    model_path = resolve_checkpoint_path("distilbert-baseline")
    if not model_path.exists():
        return None, None
    tokenizer = DistilBertTokenizerFast.from_pretrained(model_path)
    model = DistilBertForQuestionAnswering.from_pretrained(model_path)
    model.eval()
    if torch.cuda.is_available():
        model = model.cuda()
    return model, tokenizer


@st.cache_resource
def load_qwen(model_path_str: str, label: str, params: str):
    from transformers import AutoModelForCausalLM, AutoTokenizer
    model_path = Path(model_path_str)
    if not model_path.exists():
        return None, None, label, params, model_path_str
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
    return model, tokenizer, label, params, str(model_path)


# ─────────────────────────────────────────────
# Pipeline helpers
# ─────────────────────────────────────────────

def get_subgraph_and_chunks(question, entity, adjacency, index, corpus, hops=2, k=5, max_triples=20):
    from subgraph import get_subgraph
    from serialize import serialize_triples
    from embedder import embed_single

    subgraph = []
    graph_text = ""
    if entity and entity in adjacency:
        subgraph = get_subgraph(entity, adjacency, hops=hops, max_triples=max_triples)
        graph_text = serialize_triples(subgraph, style="arrow")

    q_vec = embed_single(question).astype(np.float32)
    scores, indices = index.search(q_vec, k)
    chunks = [corpus[i] for i in indices[0] if i < len(corpus)]

    return subgraph, graph_text, chunks


def run_distilbert(question, graph_text, chunks, model, tokenizer):
    if model is None:
        return "Model not loaded", 0.0

    context = graph_text + " " + " ".join(chunks)
    context = context[:1500]

    device = next(model.parameters()).device
    inputs = tokenizer(
        question, context,
        max_length=512, truncation="only_second",
        padding="max_length", return_tensors="pt"
    ).to(device)

    t0 = time.time()
    with torch.no_grad():
        outputs = model(**inputs)
    latency = (time.time() - t0) * 1000

    start = outputs.start_logits.argmax().item()
    end = outputs.end_logits.argmax().item()
    if end < start:
        end = start
    tokens = inputs['input_ids'][0][start:end+1]
    answer = tokenizer.decode(tokens, skip_special_tokens=True).strip()

    return answer if answer else "(no answer found)", latency


def run_qwen(question, graph_text, chunks, model, tokenizer, trace_output=False):
    if model is None:
        return "Model not loaded", 0.0

    from entity_extract import clean_question

    parts = []
    if graph_text:
        parts.append(f"Knowledge Graph:\n{graph_text}")
    if chunks:
        chunks_text = "\n".join(f"- {c}" for c in chunks)
        parts.append(f"Retrieved Context:\n{chunks_text}")
    parts.append(f"Question: {clean_question(question)}")
    input_text = "\n\n".join(parts)

    messages = [
        {
            "role": "system",
            "content": (
                "Answer the question using the retrieved context and knowledge graph. "
                "First list the supporting evidence from the graph or retrieved context, "
                "then write 'Final answer:' followed by the answer entity or entities separated by |."
                if trace_output else
                "Answer the question using the retrieved context and knowledge graph. Return only the answer entity or entities separated by |."
            )
        },
        {"role": "user", "content": input_text},
    ]
    prompt = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )

    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

    t0 = time.time()
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=96 if trace_output else 48,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
        )
    latency = (time.time() - t0) * 1000

    new_tokens = outputs[0][inputs['input_ids'].shape[1]:]
    answer = tokenizer.decode(new_tokens, skip_special_tokens=True).strip()

    return answer if answer else "(no answer found)", latency


# ─────────────────────────────────────────────
# Session state init
# ─────────────────────────────────────────────

if "question" not in st.session_state:
    st.session_state.question = "who directed [Inception]"

if "results" not in st.session_state:
    st.session_state.results = None


# ─────────────────────────────────────────────
# Header
# ─────────────────────────────────────────────

st.markdown('<div class="main-title">Pocket GraphRAG</div>', unsafe_allow_html=True)
st.markdown('<div class="subtitle">BASE 3B  ↔  GRAPHRAG GOLD  ↔  GRAPHRAG HYBRID</div>', unsafe_allow_html=True)
st.markdown(
    """
This app combines live GraphRAG question answering with saved benchmark and error-analysis artifacts.
The project compares a base Qwen model, gold-supervised GraphRAG students, and hybrid trace-supervised students
to study not only what performs better, but why errors happen across 1-hop, 2-hop, and 3-hop QA.
"""
)

qwen_options = [row for row in available_qwen_models() if row["path"].exists()]
default_qwen_index = next(
    (i for i, row in enumerate(qwen_options) if row["label"] == "Qwen2.5-1.5B GraphRAG Hybrid"),
    0,
)
selected_qwen_label = st.selectbox(
    "Live student checkpoint",
    [row["label"] for row in qwen_options] if qwen_options else ["No Qwen checkpoint found"],
    index=default_qwen_index if qwen_options else 0,
    disabled=not qwen_options,
)
selected_qwen = next((row for row in qwen_options if row["label"] == selected_qwen_label), None)

# ─────────────────────────────────────────────
# Load models
# ─────────────────────────────────────────────

with st.spinner("Loading models and index..."):
    try:
        adjacency, faiss_index, corpus = load_kb_and_index()
        kb_ok = True
    except Exception as e:
        kb_ok = False
        st.error(f"KB/Index load failed: {e}")

    db_model, db_tokenizer = load_distilbert()
    if selected_qwen:
        qwen_model, qwen_tokenizer, qwen_label, qwen_params, qwen_checkpoint = load_qwen(
            str(selected_qwen["path"]),
            selected_qwen["label"],
            selected_qwen["params"],
        )
    else:
        qwen_model, qwen_tokenizer, qwen_label, qwen_params, qwen_checkpoint = (
            None, None, "Qwen checkpoint", "n/a", ""
        )

col_s1, col_s2, col_s3 = st.columns(3)
with col_s1:
    st.markdown(
        f'<div class="{"status-ok" if kb_ok else "status-err"}">{"● KB + FAISS ready" if kb_ok else "✗ KB/FAISS failed"}</div>',
        unsafe_allow_html=True
    )
with col_s2:
    st.markdown(
        f'<div class="{"status-ok" if db_model else "status-err"}">{"● DistilBERT ready" if db_model else "✗ DistilBERT not found"}</div>',
        unsafe_allow_html=True
    )
with col_s3:
    st.markdown(
        f'<div class="{"status-ok" if qwen_model else "status-err"}">{"● " + qwen_label if qwen_model else "✗ " + qwen_label + " not found"}</div>',
        unsafe_allow_html=True
    )

st.markdown('<hr class="divider">', unsafe_allow_html=True)

# ─────────────────────────────────────────────
# Sample questions
# ─────────────────────────────────────────────

SAMPLES = [
    {"hop": 1, "question": "who directed [Inception]"},
    {"hop": 1, "question": "what genre is [Forrest Gump]"},
    {"hop": 2, "question": "what movies are directed by the director of [The Matrix]"},
    {"hop": 2, "question": "who acted in films directed by the director of [Inception]"},
    {"hop": 3, "question": "what genres are the movies written by the writer of [The Matrix]"},
    {"hop": 3, "question": "who starred in the films directed by the writer of [Titanic]"},
]
SAMPLE_HOPS = {item["question"]: item["hop"] for item in SAMPLES}

st.markdown("**Try a sample question (2 from each hop):**")

sample_cols = st.columns(3)
for i, sample in enumerate(SAMPLES):
    with sample_cols[i % 3]:
        st.markdown(
            f'<div class="sample-hop-label hop{sample["hop"]}">{sample["hop"]}-hop</div>',
            unsafe_allow_html=True,
        )
        label = sample["question"] if len(sample["question"]) <= 42 else sample["question"][:39] + "..."
        if st.button(label, key=f"sample_{i}", use_column_width=True):
            st.session_state.question = sample["question"]
            st.session_state.results = None
            st.rerun()

st.markdown('<hr class="divider">', unsafe_allow_html=True)
st.markdown('<div class="section-label">Saved benchmark and error-analysis overview</div>', unsafe_allow_html=True)

overview_metrics = load_metrics_table()
overview_trace = load_trace_summary_table()
overview_breakdown = load_answer_breakdown_table()

if not overview_metrics.empty:
    st.markdown("**Primary comparison: Base vs Gold vs Hybrid (3B family)**")
    overview_primary = overview_metrics[overview_metrics["model"].isin([
        "Qwen2.5-3B Instruct base",
        "Qwen2.5-3B GraphRAG Gold",
        "Qwen2.5-3B GraphRAG Hybrid",
    ])].copy()
    if not overview_primary.empty:
        overview_primary = overview_primary[[
            "model", "training", "retrieval", "1hop_EM", "2hop_EM", "3hop_EM",
            "overall_EM", "1hop_F1", "2hop_F1", "3hop_F1", "overall_F1"
        ]].rename(columns={
            "model": "Model",
            "training": "Training",
            "retrieval": "Retrieval",
            "1hop_EM": "1-hop EM",
            "2hop_EM": "2-hop EM",
            "3hop_EM": "3-hop EM",
            "overall_EM": "Overall EM",
            "1hop_F1": "1-hop F1",
            "2hop_F1": "2-hop F1",
            "3hop_F1": "3-hop F1",
            "overall_F1": "Overall F1",
        })
        st.dataframe(overview_primary, use_column_width=True, hide_index=True)

if not overview_trace.empty:
    st.markdown("**Failure-mode stats by hop**")
    overview_trace_view = overview_trace[overview_trace["dataset"].isin(["GraphRAG Hybrid", "GraphRAG Gold", "RAG Gold"])].copy()
    overview_trace_view = overview_trace_view[[
        "dataset", "hop", "avg_gold_answer_count", "gold_all_in_context_rate",
        "gold_any_in_context_rate", "avg_evidence_support_ratio",
        "avg_grounded_trace_gold_answer_coverage_rate", "avg_grounded_trace_compression_gap"
    ]].rename(columns={
        "dataset": "Dataset",
        "hop": "Hop",
        "avg_gold_answer_count": "Avg gold answers",
        "gold_all_in_context_rate": "All gold in context",
        "gold_any_in_context_rate": "Any gold in context",
        "avg_evidence_support_ratio": "Evidence support ratio",
        "avg_grounded_trace_gold_answer_coverage_rate": "Grounded trace coverage",
        "avg_grounded_trace_compression_gap": "Grounded compression gap",
    })
    st.dataframe(overview_trace_view, use_column_width=True, hide_index=True)

if not overview_breakdown.empty:
    st.markdown("**Saved answer-set error breakdown for Hybrid 3B**")
    overview_hybrid = overview_breakdown[overview_breakdown["model"] == "Qwen2.5-3B GraphRAG Hybrid"].copy()
    if not overview_hybrid.empty:
        overview_hybrid = overview_hybrid[[
            "answer_set_error", "count", "rate", "avg_example_f1",
            "partial_overlap_rate", "exact_miss_rate"
        ]].rename(columns={
            "answer_set_error": "Answer-set error",
            "count": "Count",
            "rate": "Rate",
            "avg_example_f1": "Avg example F1",
            "partial_overlap_rate": "Partial-overlap rate",
            "exact_miss_rate": "Exact-miss rate",
        })
        st.dataframe(overview_hybrid, use_column_width=True, hide_index=True)

overview_left, overview_right = st.columns(2)
with overview_left:
    for title, path in [
        ("EM by hop", Path("results/all_em_by_hop.png")),
        ("F1 by hop", Path("results/all_f1_by_hop.png")),
        ("Latency", Path("results/all_latency.png")),
    ]:
        if path.exists():
            st.markdown(f"**{title}**")
            st.image(str(path), use_column_width=True)
with overview_right:
    for title, path in [
        ("Retrieval coverage by hop", Path("results/error_analysis/retrieval_coverage_by_hop.png")),
        ("Answer burden by hop", Path("results/error_analysis/answer_burden_by_hop.png")),
        ("Teacher trace compression gap", Path("results/error_analysis/teacher_trace_compression_gap.png")),
        ("Answer-set error breakdown", Path("results/error_analysis/answer_set_error_breakdown.png")),
    ]:
        if path.exists():
            st.markdown(f"**{title}**")
            st.image(str(path), use_column_width=True)

# ─────────────────────────────────────────────
# Input form
# ─────────────────────────────────────────────

with st.form("query_form", clear_on_submit=False):
    question_input = st.text_input(
        "Question (use [brackets] around the topic entity)",
        value=st.session_state.question,
        placeholder="e.g. what movies did [Tom Hanks] star in",
    )
    trace_output = st.toggle(
        "Ask student to show evidence trace + final answer",
        value=bool(selected_qwen and selected_qwen.get("trace_native")),
        help="Hybrid checkpoints are the best fit for this mode.",
    )

    submitted = st.form_submit_button("Run pipeline →", type="primary")

if submitted:
    st.session_state.question = question_input
    st.session_state.results = None  # clear old results

# ─────────────────────────────────────────────
# Run pipeline
# ─────────────────────────────────────────────

if submitted and st.session_state.question and kb_ok:
    from entity_extract import extract_topic_entity

    question = st.session_state.question
    entity = extract_topic_entity(question)
    hops = SAMPLE_HOPS.get(question, 2)
    k = 5
    max_triples = 20

    with st.spinner("Running pipeline..."):
        subgraph, graph_text, chunks = get_subgraph_and_chunks(
            question, entity, adjacency, faiss_index, corpus,
            hops=hops, k=k, max_triples=max_triples
        )
        db_answer, db_latency = run_distilbert(
            question, graph_text, chunks, db_model, db_tokenizer
        )
        qwen_answer, qwen_latency = run_qwen(
            question, graph_text, chunks, qwen_model, qwen_tokenizer, trace_output=trace_output
        )

    # Store results in session state
    st.session_state.results = {
        "question": question,
        "entity": entity,
        "subgraph": subgraph,
        "graph_text": graph_text,
        "chunks": chunks,
        "db_answer": db_answer,
        "db_latency": db_latency,
        "qwen_answer": qwen_answer,
        "qwen_latency": qwen_latency,
        "qwen_label": qwen_label,
        "qwen_params": qwen_params,
        "qwen_checkpoint": qwen_checkpoint,
        "trace_output": trace_output,
        "hops": hops,
        "max_triples": max_triples,
    }

# ─────────────────────────────────────────────
# Display results (from session state)
# ─────────────────────────────────────────────

if st.session_state.results:
    r = st.session_state.results
    qwen_trace, qwen_final_answer = split_trace_and_answer(r.get("qwen_answer", ""))
    qwen_display_answer = qwen_final_answer or r.get("qwen_answer", "")
    selected_metrics = metrics_for_model(selected_qwen["metrics_label"]) if selected_qwen else {}

    if r["entity"]:
        st.markdown(
            f'<div class="entity-tag">Entity: {r["entity"]}</div>'
            f'<span class="hop-badge">{r["hops"]}-hop subgraph · {len(r["subgraph"])} triples</span>',
            unsafe_allow_html=True
        )

    st.markdown('<hr class="divider">', unsafe_allow_html=True)

    # Side-by-side model answers
    left, right = st.columns(2)

    with left:
        st.markdown('''
        <div class="model-card">
            <div class="model-header distilbert-header">▸ DISTILBERT BASELINE</div>
        ''', unsafe_allow_html=True)

        st.markdown(f'''
        <div class="answer-box distilbert-answer">
            <div class="answer-text">{r["db_answer"]}</div>
        </div>
        <div class="metric-row">
            <div class="metric-pill">latency <span>{r["db_latency"]:.0f}ms</span></div>
            <div class="metric-pill">method <span>span extraction</span></div>
            <div class="metric-pill">params <span>66M</span></div>
        </div>
        ''', unsafe_allow_html=True)

        st.markdown('</div>', unsafe_allow_html=True)

    with right:
        st.markdown(f'''
        <div class="model-card">
            <div class="model-header qwen-header">▸ {r.get("qwen_label", "Qwen2.5 student").upper()}</div>
        ''', unsafe_allow_html=True)

        st.markdown(f'''
        <div class="answer-box qwen-answer">
            <div class="answer-text">{qwen_display_answer}</div>
        </div>
        <div class="metric-row">
            <div class="metric-pill">latency <span>{r["qwen_latency"]:.0f}ms</span></div>
            <div class="metric-pill">method <span>generative</span></div>
            <div class="metric-pill">params <span>{r.get("qwen_params", "3B preferred")}</span></div>
            <div class="metric-pill">trace mode <span>{"on" if r.get("trace_output") else "off"}</span></div>
        </div>
        ''', unsafe_allow_html=True)

        if qwen_trace:
            st.markdown("**Evidence trace**")
            st.code(qwen_trace, language="text")

        st.markdown('</div>', unsafe_allow_html=True)

    st.markdown('<hr class="divider">', unsafe_allow_html=True)

    # Pipeline detail
    detail_left, detail_right = st.columns(2)

    with detail_left:
        st.markdown('<div class="section-label">Knowledge graph subgraph</div>', unsafe_allow_html=True)
        if r["subgraph"]:
            for subj, rel, obj in r["subgraph"][:r["max_triples"]]:
                st.markdown(
                    f'<div class="triple-item">'
                    f'<span class="entity">{subj}</span> '
                    f'<span class="rel">→ {rel} →</span> '
                    f'<span class="entity">{obj}</span>'
                    f'</div>',
                    unsafe_allow_html=True
                )
            if len(r["subgraph"]) > r["max_triples"]:
                st.markdown(
                    f'<div class="triple-item" style="color:#3d3d52">'
                    f'... {len(r["subgraph"]) - r["max_triples"]} more triples</div>',
                    unsafe_allow_html=True
                )
        else:
            st.markdown(
                '<div class="triple-item" style="color:#3d3d52">'
                'No subgraph found — check entity brackets</div>',
                unsafe_allow_html=True
            )

    with detail_right:
        st.markdown('<div class="section-label">Retrieved context (FAISS top-K)</div>', unsafe_allow_html=True)
        for i, chunk in enumerate(r["chunks"]):
            st.markdown(
                f'<div class="chunk-item">'
                f'<span style="color:#3d3d52">{i+1}.</span> {chunk}'
                f'</div>',
                unsafe_allow_html=True
            )

    st.markdown('<hr class="divider">', unsafe_allow_html=True)
    st.markdown('<div class="section-label">Reported stats for selected live student</div>', unsafe_allow_html=True)
    if selected_metrics:
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("1-hop EM", f'{selected_metrics.get("1hop_EM", 0):.3f}')
        c2.metric("2-hop EM", f'{selected_metrics.get("2hop_EM", 0):.3f}')
        c3.metric("3-hop EM", f'{selected_metrics.get("3hop_EM", 0):.3f}')
        overall_em = selected_metrics.get("overall_EM")
        c4.metric("Overall EM", f'{overall_em:.3f}' if pd.notna(overall_em) else "n/a")

        stats_df = pd.DataFrame([{
            "Model": selected_metrics.get("model", r.get("qwen_label")),
            "Training": selected_metrics.get("training", "n/a"),
            "Retrieval": selected_metrics.get("retrieval", "n/a"),
            "1-hop F1": selected_metrics.get("1hop_F1", "n/a"),
            "2-hop F1": selected_metrics.get("2hop_F1", "n/a"),
            "3-hop F1": selected_metrics.get("3hop_F1", "n/a"),
            "1-hop latency ms": selected_metrics.get("1hop_latency_ms", "n/a"),
            "2-hop latency ms": selected_metrics.get("2hop_latency_ms", "n/a"),
            "3-hop latency ms": selected_metrics.get("3hop_latency_ms", "n/a"),
            "n_samples": selected_metrics.get("n_samples", "reported benchmark"),
        }])
        st.dataframe(stats_df, use_column_width=True, hide_index=True)
    else:
        st.info("No stored benchmark row was found for the selected checkpoint in results/error_analysis/model_metrics_all.csv.")

    st.markdown('<hr class="divider">', unsafe_allow_html=True)
    st.markdown('<div class="section-label">Primary comparison: Base vs Gold vs Hybrid (3B family)</div>', unsafe_allow_html=True)
    metrics_df = load_metrics_table()
    primary_rows = metrics_df[metrics_df["model"].isin([
        "Qwen2.5-3B Instruct base",
        "Qwen2.5-3B GraphRAG Gold",
        "Qwen2.5-3B GraphRAG Hybrid",
    ])].copy() if not metrics_df.empty else pd.DataFrame()

    if not primary_rows.empty:
        primary_rows = primary_rows[[
            "model", "training", "retrieval", "1hop_EM", "2hop_EM", "3hop_EM",
            "overall_EM", "1hop_F1", "2hop_F1", "3hop_F1", "overall_F1"
        ]].rename(columns={
            "model": "Model",
            "training": "Training",
            "retrieval": "Retrieval",
            "1hop_EM": "1-hop EM",
            "2hop_EM": "2-hop EM",
            "3hop_EM": "3-hop EM",
            "overall_EM": "Overall EM",
            "1hop_F1": "1-hop F1",
            "2hop_F1": "2-hop F1",
            "3hop_F1": "3-hop F1",
            "overall_F1": "Overall F1",
        })
        st.dataframe(primary_rows, use_column_width=True, hide_index=True)
    else:
        st.info("Could not load the base/gold/hybrid comparison rows from results/error_analysis/model_metrics_all.csv.")

    st.markdown('<hr class="divider">', unsafe_allow_html=True)
    st.markdown('<div class="section-label">Failure-mode stats by hop (from saved analysis artifacts)</div>', unsafe_allow_html=True)
    trace_df = load_trace_summary_table()
    if not trace_df.empty:
        trace_view = trace_df[trace_df["dataset"].isin(["GraphRAG Hybrid", "GraphRAG Gold", "RAG Gold"])].copy()
        trace_view = trace_view[[
            "dataset", "hop", "avg_gold_answer_count", "gold_all_in_context_rate",
            "gold_any_in_context_rate", "avg_evidence_support_ratio",
            "avg_grounded_trace_gold_answer_coverage_rate",
            "avg_grounded_trace_compression_gap", "direct_answer_without_grounded_trace_rate"
        ]].rename(columns={
            "dataset": "Dataset",
            "hop": "Hop",
            "avg_gold_answer_count": "Avg gold answers",
            "gold_all_in_context_rate": "All gold in context",
            "gold_any_in_context_rate": "Any gold in context",
            "avg_evidence_support_ratio": "Evidence support ratio",
            "avg_grounded_trace_gold_answer_coverage_rate": "Grounded trace coverage",
            "avg_grounded_trace_compression_gap": "Grounded compression gap",
            "direct_answer_without_grounded_trace_rate": "Direct answer w/o grounded trace",
        })
        st.dataframe(trace_view, use_column_width=True, hide_index=True)
    else:
        st.info("No hop-level failure-mode summary file was found in results/error_analysis.")

    if selected_metrics.get("model") == "Qwen2.5-3B GraphRAG Hybrid":
        breakdown_df = load_answer_breakdown_table()
        hybrid_breakdown = breakdown_df[breakdown_df["model"] == "Qwen2.5-3B GraphRAG Hybrid"].copy() if not breakdown_df.empty else pd.DataFrame()
        if not hybrid_breakdown.empty:
            st.markdown("**Saved prediction error breakdown for Hybrid 3B**")
            hybrid_breakdown = hybrid_breakdown[[
                "answer_set_error", "count", "rate", "avg_example_f1",
                "partial_overlap_rate", "exact_miss_rate", "avg_gold_answer_count",
                "avg_pred_answer_count"
            ]].rename(columns={
                "answer_set_error": "Answer-set error",
                "count": "Count",
                "rate": "Rate",
                "avg_example_f1": "Avg example F1",
                "partial_overlap_rate": "Partial-overlap rate",
                "exact_miss_rate": "Exact-miss rate",
                "avg_gold_answer_count": "Avg gold count",
                "avg_pred_answer_count": "Avg pred count",
            })
            st.dataframe(hybrid_breakdown, use_column_width=True, hide_index=True)

    st.markdown('<hr class="divider">', unsafe_allow_html=True)
    st.markdown('<div class="section-label">Saved evaluation visuals</div>', unsafe_allow_html=True)
    perf_left, perf_right = st.columns(2)
    with perf_left:
        perf_paths = [
            ("EM by hop", Path("results/all_em_by_hop.png")),
            ("F1 by hop", Path("results/all_f1_by_hop.png")),
            ("Latency", Path("results/all_latency.png")),
        ]
        for title, path in perf_paths:
            if path.exists():
                st.markdown(f"**{title}**")
                st.image(str(path), use_column_width=True)
    with perf_right:
        analysis_paths = [
            ("Retrieval coverage by hop", Path("results/error_analysis/retrieval_coverage_by_hop.png")),
            ("Answer burden by hop", Path("results/error_analysis/answer_burden_by_hop.png")),
            ("Teacher trace compression gap", Path("results/error_analysis/teacher_trace_compression_gap.png")),
            ("Answer-set error breakdown", Path("results/error_analysis/answer_set_error_breakdown.png")),
        ]
        for title, path in analysis_paths:
            if path.exists():
                st.markdown(f"**{title}**")
                st.image(str(path), use_column_width=True)

    # Benchmark table
    st.markdown('<hr class="divider">', unsafe_allow_html=True)
    st.markdown('<div class="section-label">Benchmark results (test set · 200 samples per hop)</div>', unsafe_allow_html=True)

    import pandas as pd
    bench_data = {
        "Model": ["DistilBERT baseline", "Qwen2.5-3B GraphRAG Gold"],
        "1-hop EM": [0.449, 0.832],
        "2-hop EM": [0.649, 0.586],
        "3-hop EM": [0.059, 0.050],
        "Overall EM": ["-", 0.489],
        "Avg latency": ["~5ms", "~600ms"],
        "Params": ["66M", "3B"],
    }
    df = pd.DataFrame(bench_data)
    st.dataframe(df, use_column_width=True, hide_index=True)

elif submitted and not kb_ok:
    st.error("KB or FAISS index not loaded. Check data/raw/kb.txt and data/faiss/")
