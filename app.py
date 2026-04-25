"""
app.py — Pocket GraphRAG Demo
Side-by-side comparison: DistilBERT baseline vs Qwen2.5 student

Run with:
    streamlit run app.py
"""

import sys
import time
from pathlib import Path

import streamlit as st
import torch
import numpy as np

sys.path.insert(0, "src/data")
sys.path.insert(0, "src/graph")
sys.path.insert(0, "src/retrieval")

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
    model_path = "checkpoints/distilbert-baseline"
    if not Path(model_path).exists():
        return None, None
    tokenizer = DistilBertTokenizerFast.from_pretrained(model_path)
    model = DistilBertForQuestionAnswering.from_pretrained(model_path)
    model.eval()
    if torch.cuda.is_available():
        model = model.cuda()
    return model, tokenizer


@st.cache_resource
def load_qwen():
    from transformers import AutoModelForCausalLM, AutoTokenizer

    candidates = [
        ("Qwen2.5-3B GraphRAG Gold", "checkpoints/qwen2.5-3b-graphrag-gold", "3B (LoRA)"),
        ("Qwen2.5-3B GraphRAG Hybrid", "checkpoints/qwen2.5-3b-graphrag-hybrid", "3B (LoRA + traces)"),
        ("Qwen2.5-1.5B GraphRAG Gold", "checkpoints/qwen2.5-1.5b-graphrag-gold", "1.5B (LoRA)"),
        ("Qwen2.5-1.5B GraphRAG Hybrid", "checkpoints/qwen2.5-1.5b-graphrag-hybrid", "1.5B (LoRA + traces)"),
    ]

    selected = next(
        ((label, path, params) for label, path, params in candidates if Path(path).exists()),
        None,
    )
    if selected is None:
        return None, None, "Qwen2.5-3B student preferred", "3B preferred", ""

    label, model_path, params = selected
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.float16,
        device_map="auto",
        trust_remote_code=True,
    )
    model.eval()
    return model, tokenizer, label, params, model_path


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


def run_qwen(question, graph_text, chunks, model, tokenizer):
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
            "content": "Answer the question using the retrieved context and knowledge graph. Return only the answer entity or entities separated by |."
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
            max_new_tokens=64,
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
st.markdown('<div class="subtitle">DISTILBERT BASELINE  ↔  QWEN2.5-3B STUDENT (PREFERRED)</div>', unsafe_allow_html=True)

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
    qwen_model, qwen_tokenizer, qwen_label, qwen_params, qwen_checkpoint = load_qwen()

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
    "who directed [Inception]",
    "what genre is [Forrest Gump]",
    "what movies did [Tom Hanks] star in",
    "what is the rating of [The Dark Knight]",
    "who wrote [Pulp Fiction]",
    "what movies are directed by the director of [The Matrix]",
    "who acted in films directed by the director of [Inception]",
    "what is the release year of [Titanic]",
]

st.markdown("**Try a sample question:**")

sample_cols = st.columns(4)
for i, sample in enumerate(SAMPLES[:4]):
    with sample_cols[i]:
        label = sample if len(sample) <= 36 else sample[:33] + "..."
        if st.button(label, key=f"sample_{i}"):
            st.session_state.question = sample
            st.session_state.results = None
            st.rerun()

sample_cols2 = st.columns(4)
for i, sample in enumerate(SAMPLES[4:]):
    with sample_cols2[i]:
        label = sample if len(sample) <= 36 else sample[:33] + "..."
        if st.button(label, key=f"sample2_{i}"):
            st.session_state.question = sample
            st.session_state.results = None
            st.rerun()

# ─────────────────────────────────────────────
# Input form
# ─────────────────────────────────────────────

with st.form("query_form", clear_on_submit=False):
    question_input = st.text_input(
        "Question (use [brackets] around the topic entity)",
        value=st.session_state.question,
        placeholder="e.g. what movies did [Tom Hanks] star in",
    )

    col_s1, col_s2, col_s3 = st.columns(3)
    with col_s1:
        hops = st.selectbox("Graph hops", [1, 2, 3], index=1)
    with col_s2:
        k = st.selectbox("Retrieved chunks (K)", [3, 5, 10], index=1)
    with col_s3:
        max_triples = st.selectbox("Max graph triples", [10, 20, 30, 50], index=1)

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

    with st.spinner("Running pipeline..."):
        subgraph, graph_text, chunks = get_subgraph_and_chunks(
            question, entity, adjacency, faiss_index, corpus,
            hops=hops, k=k, max_triples=max_triples
        )
        db_answer, db_latency = run_distilbert(
            question, graph_text, chunks, db_model, db_tokenizer
        )
        qwen_answer, qwen_latency = run_qwen(
            question, graph_text, chunks, qwen_model, qwen_tokenizer
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
        "hops": hops,
        "max_triples": max_triples,
    }

# ─────────────────────────────────────────────
# Display results (from session state)
# ─────────────────────────────────────────────

if st.session_state.results:
    r = st.session_state.results

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
            <div class="answer-text">{r["qwen_answer"]}</div>
        </div>
        <div class="metric-row">
            <div class="metric-pill">latency <span>{r["qwen_latency"]:.0f}ms</span></div>
            <div class="metric-pill">method <span>generative</span></div>
            <div class="metric-pill">params <span>{r.get("qwen_params", "3B preferred")}</span></div>
        </div>
        ''', unsafe_allow_html=True)

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
    st.dataframe(df, use_container_width=True, hide_index=True)

elif submitted and not kb_ok:
    st.error("KB or FAISS index not loaded. Check data/raw/kb.txt and data/faiss/")
