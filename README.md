# Pocket GraphRAG — Stage 2 & 3: RAG + GraphRAG Pipeline

## Branch: `post-baseline`

This branch builds on the `baseline` branch and adds FAISS semantic retrieval, N-hop graph traversal, GPT-4o teacher generation, and the instruction set that will be used to fine-tune the Qwen2.5 student model.

---

## What this branch adds

```
pocket-graphrag/
├── data/
│   ├── faiss/
│   │   ├── index.bin               # FAISS vector index (auto-generated)
│   │   └── corpus.pkl              # KB triple corpus (auto-generated)
│   └── processed/
│       ├── cache/                  # Tokenized dataset cache (from baseline)
│       ├── teacher_outputs/        # GPT-4o generated answers (checkpointed)
│       └── train_instruction.json  # Final instruction set for Qwen2.5
├── src/
│   ├── retrieval/
│   │   ├── embedder.py             # sentence-transformers wrapper
│   │   ├── faiss_index.py          # Build and query FAISS index
│   │   ├── build_index.py          # One-time index build script
│   │   └── retrieve.py             # Full RAG pipeline (entity → graph + FAISS → prompt)
│   ├── graph/
│   │   ├── entity_extract.py       # Extract [bracketed entity] from question
│   │   ├── subgraph.py             # N-hop graph traversal
│   │   └── serialize.py            # Triples → text for prompts
│   └── teacher/
│       ├── generate.py             # Call GPT-4o, checkpoint every 10 examples
│       └── format_pairs.py         # Convert teacher outputs → instruction JSON
└── README.md
```

---

## Setup

Make sure you have everything from the `baseline` branch set up first, then:

```bash
pip install sentence-transformers faiss-cpu openai
```

---

## Stage 2 — Build the FAISS Index

The retrieval corpus is every triple in `kb.txt` serialised to a sentence:
```
The Matrix directed_by Wachowski Sisters
The Matrix has_genre Sci-Fi
Tom Hanks starred_actors Cast Away
...
```
134,741 triples → 133,582 unique sentences (after deduplication) → encoded into 384-dim vectors → stored in FAISS flat index.

```bash
# Run once — saves index to data/faiss/
python src/retrieval/build_index.py --kb_path data/raw/kb.txt
```

Takes ~2-3 minutes. Index is saved to disk and reloaded instantly on subsequent runs.

**Test retrieval:**
```bash
python src/retrieval/retrieve.py
```

Expected output for `"Who directed The Matrix?"`:
```
Entity: The Matrix
Subgraph: The Matrix → directed_by → Wachowski Sisters
Retrieved: The Matrix directed_by Wachowski Sisters
```

---

## How the retrieval pipeline works

For each question, the pipeline runs 3 steps in parallel:

```
Question: "What movies did [Tom Hanks] star in?"
        ↓
1. Entity extraction    →  topic_entity = "Tom Hanks"
        ↓
2. Graph traversal      →  subgraph: {Cast Away → starred_actors → Tom Hanks, ...}
        ↓
3. FAISS retrieval      →  top-5 KB sentences most similar to the question
        ↓
4. Build prompt         →  question + subgraph + retrieved chunks → GPT-4o
```

**Three retrieval modes** (controlled by `mode` argument):

| Mode | Uses graph | Uses FAISS | Used for |
|------|-----------|-----------|---------|
| `rag` | No | Yes | Stage 2 ablation |
| `graph` | Yes | No | Stage 3 ablation |
| `graphrag` | Yes | Yes | Stage 3 (default) |

---

## Stage 3 — GPT-4o Teacher Generation

The teacher model (GPT-4o) receives a structured prompt containing the question, the knowledge graph subgraph, and the top-K retrieved context chunks. It generates a clean, accurate answer that becomes the training target for the student model.

**What GPT-4o receives:**
```
You are a question answering assistant. Answer the question based on the
provided context. Be concise — give only the answer entity or entities,
separated by | if there are multiple.

Knowledge Graph:
Cast Away → starred_actors → Tom Hanks
Philadelphia → starred_actors → Tom Hanks
Forrest Gump → starred_actors → Tom Hanks

Retrieved Context:
- Cast Away starred_actors Tom Hanks
- Philadelphia starred_actors Tom Hanks

Question: What movies did Tom Hanks star in?

Answer:
```

**GPT-4o returns:** `Cast Away | Philadelphia | Forrest Gump`

This (prompt → answer) pair is saved as an instruction tuple:
```json
{
  "instruction": "Given the context and knowledge graph, answer the question.",
  "input": "Knowledge Graph:\n...\n\nRetrieved Context:\n...\n\nQuestion: ...",
  "output": "Cast Away | Philadelphia | Forrest Gump"
}
```

**Run teacher generation:**
```bash
# Set your OpenAI API key first
export OPENAI_API_KEY=sk-...       # Linux/Mac
$env:OPENAI_API_KEY="sk-..."       # Windows PowerShell

python src/teacher/generate.py \
    --data_dir data/raw \
    --kb_path data/raw/kb.txt \
    --output_path data/processed/teacher_outputs/outputs.json \
    --max_samples 200 \
    --hops 1
```

**Cost estimate:**

| Samples | Estimated cost |
|---------|---------------|
| 200 (pipeline test) | ~$1 |
| 1,500 (full training) | ~$7-10 |
| Ablations (graph only, RAG only, K values) | ~$5 |
| **Total** | **~$15-20** |

Outputs are checkpointed every 10 examples so generation can be resumed if interrupted.

---

## Instruction set format

The final instruction set saved to `data/processed/train_instruction.json` follows the LlamaFactory format:

```json
[
  {
    "instruction": "Given the context and knowledge graph, answer the question.",
    "input": "Knowledge Graph:\nThe Matrix → directed_by → Wachowski Sisters\n\nRetrieved Context:\n- The Matrix directed_by Wachowski Sisters\n\nQuestion: Who directed The Matrix?",
    "output": "Wachowski Sisters"
  },
  ...
]
```

Each entry captures:
- The full graph + retrieval context the model saw
- The gold-quality answer GPT-4o produced
- The reasoning style of a large teacher model

This is the dataset used to fine-tune Qwen2.5 via LoRA in Stage 4.

---

## Ablation experiments (this branch)

| Experiment | What changes | Purpose |
|-----------|-------------|---------|
| RAG only | No graph, FAISS only | Measure value of graph |
| Graph only | No FAISS, graph only | Measure value of retrieval |
| RAG + Graph | Both (default) | Full GraphRAG |
| K=3,5,10,15 | Number of retrieved chunks | Find optimal K |

---

## Stage 2 & 3 expected results

These are targets — actual numbers will be logged to WandB.

| Stage | Model | 1-hop EM | 2-hop EM | 3-hop EM | Latency |
|-------|-------|--------:|--------:|--------:|--------:|
| 1 — Baseline | DistilBERT | 0.449 | 0.649 | 0.059 | 5ms |
| 2 — RAG | GPT-4o | ~0.85 | ~0.70 | ~0.50 | ~1000ms |
| 3 — GraphRAG | GPT-4o | ~0.90 | ~0.80 | ~0.70 | ~1500ms |
| 4 — Student | Qwen2.5-1.5B | ~0.85 | ~0.75 | ~0.65 | ~100ms |

The key result is Stage 4 — a tiny local model matching GPT-4o quality at 15x lower latency and zero API cost.

---

## Tech stack (Stage 2 & 3)

| Component | Tool |
|-----------|------|
| Embeddings | `sentence-transformers/all-MiniLM-L6-v2` |
| Vector retrieval | FAISS flat index (IndexFlatIP, cosine similarity) |
| Graph traversal | Bidirectional adjacency dict (custom) |
| Teacher model | GPT-4o via OpenAI API |
| Prompt format | LlamaFactory instruction format |
| Experiment tracking | Weights & Biases |

---

## Next stage

See branch `student-training` for Qwen2.5 LoRA fine-tuning via LlamaFactory.
# Pocket GraphRAG - MetaQA RAG/GraphRAG Project

## Current Deliverables

This repository now includes three main project-facing artifacts:

- Full visual MetaQA EDA notebook: `notebooks/01_metaqa_full_eda.ipynb`
- Streamlit dashboard/frontend: `app/streamlit_app.py`
- Setup and usage documentation: `docs/project_usage.md`

Install dependencies:

```bash
pip install -r requirements.txt
```

Run the dashboard:

```bash
streamlit run app/streamlit_app.py
```

Open the EDA notebook:

```bash
jupyter notebook notebooks/01_metaqa_full_eda.ipynb
```

The current raw dataset layout is expected under `src/data/raw/`, including `kb.txt`, `entity/kb_entity_dict.txt`, and `1-hop`, `2-hop`, `3-hop` QA folders. See `docs/project_usage.md` for the full workflow, method descriptions, limitations, and interpretation guidance.

---
