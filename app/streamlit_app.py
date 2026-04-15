"""
Streamlit dashboard for testing MetaQA systems.

Run:
    streamlit run app/streamlit_app.py

The MVP supports retrieval/graph coverage evaluation without needing trained
Qwen checkpoints. Optional model-backed modes are available when checkpoints,
API keys, or Ollama models are configured.
"""

from __future__ import annotations

import random
import sys
import time
from pathlib import Path

import pandas as pd
import streamlit as st


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
for module_dir in ("data", "evaluation", "graph", "models", "retrieval"):
    sys.path.insert(0, str(SRC_DIR / module_dir))

from load_kb import load_kb
from load_metaqa import load_all_splits, sample_qa_pairs
from metrics import exact_match, f1_score
from faiss_index import load_or_build_index, triples_to_corpus
from retrieve import recall_at_k, run_rag_pipeline
from serialize import serialize_triples
from subgraph import answer_in_subgraph


HOP_DIRS = {1: "1-hop", 2: "2-hop", 3: "3-hop"}


def normalize_prediction(text: str) -> str:
    if "Final answer:" in text:
        text = text.split("Final answer:", 1)[1]
    return text.strip()


def default_raw_dir() -> Path:
    return PROJECT_ROOT / "src" / "data" / "raw"


@st.cache_resource(show_spinner=False)
def load_graph_resources(raw_dir: str):
    raw = Path(raw_dir)
    triples, adjacency = load_kb(str(raw / "kb.txt"))
    corpus = triples_to_corpus(triples)
    return triples, adjacency, corpus


@st.cache_resource(show_spinner=False)
def load_faiss_resources(raw_dir: str):
    triples, _, corpus = load_graph_resources(raw_dir)
    index_path = PROJECT_ROOT / "data" / "faiss" / "index.bin"
    corpus_path = PROJECT_ROOT / "data" / "faiss" / "corpus.pkl"
    index, corpus = load_or_build_index(
        corpus,
        index_path=str(index_path),
        corpus_path=str(corpus_path),
    )
    return index, corpus


@st.cache_resource(show_spinner=False)
def load_distilbert(checkpoint_path: str):
    import torch
    from transformers import DistilBertForQuestionAnswering, DistilBertTokenizerFast

    path = checkpoint_path.strip()
    tokenizer = DistilBertTokenizerFast.from_pretrained(path)
    model = DistilBertForQuestionAnswering.from_pretrained(path)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()
    return model, tokenizer, device


@st.cache_resource(show_spinner=False)
def load_hf_causal_model(model_name_or_path: str):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
    model = AutoModelForCausalLM.from_pretrained(
        model_name_or_path,
        device_map="auto",
        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
    )
    model.eval()
    return model, tokenizer


def hop_path(raw_dir: Path, hop: int) -> Path:
    candidate = raw_dir / HOP_DIRS[hop]
    if candidate.exists():
        return candidate
    return raw_dir / f"{hop}hop"


def load_sample(raw_dir: Path, split: str, samples_per_hop: int, seed: int | None):
    rng = random.Random(seed) if seed is not None else random.Random()
    samples = []
    for hop in (1, 2, 3):
        splits = load_all_splits(str(hop_path(raw_dir, hop)))
        for item in sample_qa_pairs(splits[split], samples_per_hop, rng=rng):
            enriched = dict(item)
            enriched["hop"] = hop
            samples.append(enriched)
    return samples


def answer_with_oracle(item: dict) -> str:
    return " | ".join(item["answers"])


def answer_with_rag_oracle(item: dict, pipeline_result: dict) -> str:
    context = " ".join(pipeline_result["retrieved_chunks"]).lower()
    hits = [answer for answer in item["answers"] if answer.lower() in context]
    return " | ".join(hits)


def answer_with_graph_oracle(item: dict, pipeline_result: dict) -> str:
    entities = set()
    for subj, _, obj in pipeline_result["subgraph"]:
        entities.add(subj.lower())
        entities.add(obj.lower())
    hits = [answer for answer in item["answers"] if answer.lower() in entities]
    return " | ".join(hits)


def answer_with_distilbert(item: dict, adjacency: dict, checkpoint_path: str) -> str:
    import torch
    from baseline import build_context

    model, tokenizer, device = load_distilbert(checkpoint_path)
    context = build_context(item["topic_entity"], adjacency, hops=max(item["hop"], 2))
    encoded = tokenizer(
        item["question"],
        context,
        max_length=384,
        truncation="only_second",
        padding="max_length",
        return_tensors="pt",
    )
    encoded = {key: value.to(device) for key, value in encoded.items()}
    with torch.no_grad():
        output = model(**encoded)
    start = int(output.start_logits.argmax())
    end = int(output.end_logits.argmax())
    if end < start:
        end = start
    return tokenizer.decode(encoded["input_ids"][0][start : end + 1], skip_special_tokens=True)


def answer_with_openai(prompt: str, model_name: str, api_key: str) -> str:
    from openai import OpenAI

    client = OpenAI(api_key=api_key)
    response = client.responses.create(model=model_name, input=prompt)
    return response.output_text.strip()


def answer_with_ollama(prompt: str, model_name: str, endpoint: str) -> str:
    import requests

    response = requests.post(
        endpoint.rstrip("/") + "/api/generate",
        json={"model": model_name, "prompt": prompt, "stream": False},
        timeout=180,
    )
    response.raise_for_status()
    return response.json().get("response", "").strip()


def answer_with_hf(prompt: str, model_name_or_path: str, max_new_tokens: int) -> str:
    import torch

    model, tokenizer = load_hf_causal_model(model_name_or_path)
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    with torch.no_grad():
        output = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
        )
    generated = output[0][inputs["input_ids"].shape[1] :]
    return tokenizer.decode(generated, skip_special_tokens=True).strip()


def run_one_example(
    item: dict,
    method: str,
    adjacency: dict,
    faiss_index,
    corpus: list[str],
    k: int,
    max_triples: int,
    settings: dict,
) -> dict:
    mode = "graphrag" if "GraphRAG" in method or "Graph" in method else "rag"
    if method == "DistilBERT baseline":
        mode = "graph"

    pipeline_result = run_rag_pipeline(
        item["question"],
        adjacency,
        faiss_index,
        corpus,
        hops=item["hop"],
        k=k,
        mode=mode,
        max_triples=max_triples,
    )

    start = time.perf_counter()
    try:
        if method == "Gold oracle":
            prediction = answer_with_oracle(item)
        elif method == "RAG retrieval oracle":
            prediction = answer_with_rag_oracle(item, pipeline_result)
        elif method == "GraphRAG coverage oracle":
            prediction = answer_with_graph_oracle(item, pipeline_result)
        elif method == "DistilBERT baseline":
            prediction = answer_with_distilbert(item, adjacency, settings["distilbert_path"])
        elif method == "RAG + OpenAI":
            prediction = answer_with_openai(
                pipeline_result["prompt"],
                settings["openai_model"],
                settings["openai_api_key"],
            )
        elif method == "GraphRAG + OpenAI":
            prediction = answer_with_openai(
                pipeline_result["prompt"],
                settings["openai_model"],
                settings["openai_api_key"],
            )
        elif method == "Qwen/HF causal LM":
            prediction = answer_with_hf(
                pipeline_result["prompt"],
                settings["hf_model"],
                settings["max_new_tokens"],
            )
        elif method == "Ollama local model":
            prediction = answer_with_ollama(
                pipeline_result["prompt"],
                settings["ollama_model"],
                settings["ollama_endpoint"],
            )
        else:
            prediction = ""
        error = ""
    except Exception as exc:
        prediction = ""
        error = str(exc)

    latency_ms = (time.perf_counter() - start) * 1000
    prediction = normalize_prediction(prediction)
    em = exact_match(prediction, item["answers"])
    f1 = f1_score(prediction, item["answers"])

    return {
        "hop": item["hop"],
        "question": item["question"],
        "topic_entity": item["topic_entity"],
        "gold_answers": " | ".join(item["answers"]),
        "prediction": prediction,
        "EM": em,
        "F1": f1,
        "latency_ms": round(latency_ms, 2),
        "recall_at_k": recall_at_k(item["answers"], pipeline_result["retrieved_chunks"]),
        "answer_in_subgraph": answer_in_subgraph(item["answers"], pipeline_result["subgraph"]),
        "retrieved_chunks": pipeline_result["retrieved_chunks"],
        "subgraph": serialize_triples(pipeline_result["subgraph"], style="arrow"),
        "prompt": pipeline_result["prompt"],
        "error": error,
    }


def summarize(results: list[dict]) -> pd.DataFrame:
    df = pd.DataFrame(results)
    if df.empty:
        return pd.DataFrame()
    return (
        df.groupby("hop")
        .agg(
            examples=("question", "count"),
            EM=("EM", "mean"),
            F1=("F1", "mean"),
            recall_at_k=("recall_at_k", "mean"),
            answer_in_subgraph=("answer_in_subgraph", "mean"),
            latency_ms=("latency_ms", "mean"),
        )
        .reset_index()
    )


st.set_page_config(page_title="Pocket GraphRAG Dashboard", layout="wide")
st.title("Pocket GraphRAG MetaQA Dashboard")
st.caption("Evaluate baseline, RAG, GraphRAG, and optional local/API student models on random MetaQA samples.")

with st.sidebar:
    st.header("Data")
    raw_dir = Path(st.text_input("Raw data directory", value=str(default_raw_dir())))
    split = st.selectbox("Split", ["test", "dev", "train"], index=0)
    samples_per_hop = st.slider("Random samples per hop", 1, 50, 3)
    seed_text = st.text_input("Seed (blank = fresh random)", value="")
    seed = int(seed_text) if seed_text.strip().isdigit() else None

    st.header("Method")
    method = st.selectbox(
        "System",
        [
            "Gold oracle",
            "RAG retrieval oracle",
            "GraphRAG coverage oracle",
            "DistilBERT baseline",
            "RAG + OpenAI",
            "GraphRAG + OpenAI",
            "Qwen/HF causal LM",
            "Ollama local model",
        ],
    )
    k = st.slider("Retrieved chunks K", 1, 20, 5)
    max_triples = st.slider("Max graph triples", 5, 250, 50)

    st.header("Optional Model Settings")
    distilbert_path = st.text_input("DistilBERT checkpoint", value="checkpoints/distilbert-baseline")
    openai_model = st.text_input("OpenAI model", value="gpt-4o-mini")
    openai_api_key = st.text_input("OpenAI API key", value="", type="password")
    hf_model = st.text_input("HF/Qwen model or checkpoint", value="Qwen/Qwen2.5-1.5B-Instruct")
    max_new_tokens = st.slider("Max new tokens", 8, 512, 64)
    ollama_model = st.text_input("Ollama model", value="pocket-graphrag")
    ollama_endpoint = st.text_input("Ollama endpoint", value="http://localhost:11434")

settings = {
    "distilbert_path": distilbert_path,
    "openai_model": openai_model,
    "openai_api_key": openai_api_key,
    "hf_model": hf_model,
    "max_new_tokens": max_new_tokens,
    "ollama_model": ollama_model,
    "ollama_endpoint": ollama_endpoint,
}

if st.button("Run Evaluation", type="primary"):
    if not (raw_dir / "kb.txt").exists():
        st.error(f"Could not find kb.txt in {raw_dir}")
        st.stop()

    with st.status("Loading graph, retrieval index, and random questions...", expanded=True) as status:
        triples, adjacency, _ = load_graph_resources(str(raw_dir))
        faiss_index, corpus = load_faiss_resources(str(raw_dir))
        samples = load_sample(raw_dir, split, samples_per_hop, seed)
        status.write(f"Loaded {len(triples):,} triples")
        status.write(f"Sampled {len(samples)} questions from {split}")
        status.update(label="Running selected method...", state="running")

        results = []
        progress = st.progress(0)
        for idx, item in enumerate(samples, start=1):
            result = run_one_example(
                item,
                method,
                adjacency,
                faiss_index,
                corpus,
                k,
                max_triples,
                settings,
            )
            results.append(result)
            status.write(f"{idx}/{len(samples)} | {item['hop']}-hop | EM={result['EM']:.0f} | F1={result['F1']:.2f}")
            progress.progress(idx / len(samples))
        status.update(label="Evaluation complete", state="complete")

    results_df = pd.DataFrame(results)
    summary_df = summarize(results)

    st.subheader("Summary by Hop")
    st.dataframe(summary_df, use_container_width=True)

    st.subheader("Per-Question Results")
    st.dataframe(
        results_df[
            [
                "hop",
                "question",
                "gold_answers",
                "prediction",
                "EM",
                "F1",
                "recall_at_k",
                "answer_in_subgraph",
                "latency_ms",
                "error",
            ]
        ],
        use_container_width=True,
    )

    with st.expander("Inspect Retrieval, Subgraph, and Prompts"):
        selected = st.selectbox(
            "Question",
            options=list(range(len(results))),
            format_func=lambda i: f"{results[i]['hop']}-hop | {results[i]['question'][:100]}",
        )
        row = results[selected]
        st.markdown("**Retrieved Chunks**")
        st.write(row["retrieved_chunks"])
        st.markdown("**Subgraph**")
        st.text(row["subgraph"])
        st.markdown("**Prompt**")
        st.text(row["prompt"])
else:
    st.info("Choose a method and click Run Evaluation. Start with the oracle modes to verify retrieval and graph coverage before loading heavier models.")
