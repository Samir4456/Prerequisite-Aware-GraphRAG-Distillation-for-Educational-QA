# Pocket GraphRAG Project Usage

This document explains the current repository artifacts for MetaQA analysis, RAG/GraphRAG experimentation, dashboard testing, and dependency setup.

## 1. Data Layout

The project expects MetaQA raw files under:

```text
src/data/raw/
```

Current expected files:

```text
src/data/raw/kb.txt
src/data/raw/entity/kb_entity_dict.txt
src/data/raw/1-hop/qa_train.txt
src/data/raw/1-hop/qa_dev.txt
src/data/raw/1-hop/qa_test.txt
src/data/raw/2-hop/qa_train.txt
src/data/raw/2-hop/qa_dev.txt
src/data/raw/2-hop/qa_test.txt
src/data/raw/3-hop/qa_train.txt
src/data/raw/3-hop/qa_dev.txt
src/data/raw/3-hop/qa_test.txt
```

The loader also accepts older folder names such as `1hop`, `2hop`, and `3hop`.

## 2. Installation

Create and activate an environment, then install dependencies:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

If installing GPU-enabled PyTorch, use the command from the official PyTorch selector for your CUDA version, then run the rest of `requirements.txt`.

## 3. EDA Notebook

Notebook:

```text
notebooks/01_metaqa_full_eda.ipynb
```

Supporting module:

```text
src/eda/metaqa_analysis.py
```

The notebook covers:

- Question counts by hop and split.
- Answer count distribution.
- Question length distribution.
- Top topic entities.
- Approximate question type distribution.
- Knowledge graph node/edge/relation statistics.
- Relation frequency.
- Degree distribution.
- Top high-degree entities.
- In-degree vs out-degree.
- Sampled shortest path lengths.

Run it with:

```powershell
jupyter notebook notebooks/01_metaqa_full_eda.ipynb
```

or:

```powershell
jupyter lab
```

## 4. Dashboard

Dashboard:

```text
app/streamlit_app.py
```

Run:

```powershell
streamlit run app/streamlit_app.py
```

The dashboard randomly samples questions from 1-hop, 2-hop, and 3-hop splits and evaluates the selected method.

### MVP Methods

The MVP methods do not require trained Qwen checkpoints:

| Method | Meaning |
|---|---|
| Gold oracle | Returns the gold MetaQA answer; sanity upper bound |
| RAG retrieval oracle | Returns gold answers only if they appear in retrieved chunks; retrieval ceiling |
| GraphRAG coverage oracle | Returns gold answers only if they appear in the extracted subgraph; graph coverage ceiling |

These modes are useful before training because they show whether failures are likely caused by retrieval/subgraph coverage or by answer generation.

### Optional Model Methods

These methods require extra configuration:

| Method | Requirement |
|---|---|
| DistilBERT baseline | A local DistilBERT QA checkpoint path |
| RAG + OpenAI | OpenAI API key |
| GraphRAG + OpenAI | OpenAI API key |
| Qwen/HF causal LM | HuggingFace model ID or local checkpoint |
| Ollama local model | Running Ollama server and model name |

Use `Qwen/HF causal LM` for trained or non-trained Qwen models:

```text
Qwen/Qwen2.5-1.5B-Instruct
Qwen/Qwen2.5-3B-Instruct
path/to/fine-tuned/qwen/checkpoint
```

Use `Ollama local model` for exported local models:

```text
pocket-graphrag
qwen2.5:1.5b
```

## 5. Retrieval Index

The dashboard loads or builds a FAISS index automatically under:

```text
data/faiss/index.bin
data/faiss/corpus.pkl
```

The first run may take a few minutes because the KB triples are embedded with `sentence-transformers/all-MiniLM-L6-v2`.

To build manually:

```powershell
python src/retrieval/build_index.py --kb_path src/data/raw/kb.txt
```

Note: the legacy build script saves to `data/faiss/` by default. The dashboard uses the same cache location.

## 6. Recommended Workflow

### Step A: Run EDA

Use the notebook first. Pull report figures from:

- Dataset counts.
- Answer count distribution.
- Top topic entities.
- Relation frequency.
- Degree distribution.
- Top hub entities.
- Sampled shortest paths.

### Step B: Test Retrieval and Graph Coverage

Run the dashboard and select:

```text
RAG retrieval oracle
GraphRAG coverage oracle
```

Evaluate a small random sample first:

```text
3 to 10 questions per hop
```

These results tell you whether RAG/GraphRAG has enough evidence to answer the question.

### Step C: Test Answer Generators

Once retrieval looks reasonable, try:

```text
RAG + OpenAI
GraphRAG + OpenAI
```

Then compare:

```text
Qwen/HF causal LM with base Qwen
Qwen/HF causal LM with fine-tuned Qwen
```

### Step D: Train and Compare

Recommended comparisons:

- DistilBERT baseline.
- Zero-shot RAG.
- Zero-shot GraphRAG.
- Qwen2.5 Gold GraphRAG SFT.
- Qwen2.5 Hybrid Evidence Distillation.

Evaluate each separately by hop depth.

## 7. Interpretation Guidance

If RAG retrieval oracle is low:

```text
The vector retriever is the bottleneck.
Try higher K, better corpus text, or graph-aware retrieval.
```

If GraphRAG coverage oracle is low:

```text
The subgraph extraction is missing answers.
Try more hops, larger max_triples, or path-guided subgraph extraction.
```

If oracle coverage is high but model EM/F1 is low:

```text
The answer generator is the bottleneck.
Fine-tuning or better prompting is likely useful.
```

If 3-hop performance collapses:

```text
This is expected and should be reported separately.
Analyze whether retrieval coverage, graph coverage, or generation quality failed.
```

## 8. Current Limitations

- The dashboard's oracle modes are diagnostic ceilings, not deployable QA systems.
- HuggingFace/Qwen loading can require significant GPU memory.
- DistilBERT mode requires a previously saved checkpoint.
- OpenAI modes require an API key and network access.
- The Streamlit app builds the FAISS index on first use, which can be slow.

## 9. Deliverables Created

| File | Purpose |
|---|---|
| `notebooks/01_metaqa_full_eda.ipynb` | Visual MetaQA EDA notebook |
| `src/eda/metaqa_analysis.py` | Reusable EDA functions |
| `app/streamlit_app.py` | Dashboard/frontend for testing methods |
| `requirements.txt` | Dependency list |
| `instruction_set_strategy.md` | Gold SFT vs distillation strategy |
| `docs/project_usage.md` | Setup, EDA, dashboard, and evaluation docs |

