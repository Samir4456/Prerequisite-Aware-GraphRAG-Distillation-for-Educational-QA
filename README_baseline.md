# Pocket GraphRAG — Stage 1: DistilBERT Baseline

## Branch: `baseline`

This branch contains the complete data pipeline and DistilBERT baseline (Stage 1) for the Prerequisite-Aware GraphRAG Distillation project. Every subsequent stage must beat the numbers recorded here.

---

## What this branch contains

```
pocket-graphrag/
├── data/
│   ├── raw/                        # MetaQA dataset (not committed — download separately)
│   │   ├── kb.txt                  # Knowledge graph: 134,741 triples
│   │   ├── 1hop/
│   │   │   ├── qa_train.txt        # 96,106 train questions
│   │   │   ├── qa_dev.txt          # 9,992 dev questions
│   │   │   └── qa_test.txt         # 9,947 test questions
│   │   ├── 2hop/                   # 118,980 / 14,872 / 14,872
│   │   └── 3hop/                   # 114,196 / 14,274 / 14,274
│   └── processed/
│       └── cache/                  # Tokenized dataset cache (auto-generated)
├── src/
│   └── data/
│       ├── load_kb.py              # Parse kb.txt → bidirectional adjacency dict
│       ├── load_metaqa.py          # Load QA pairs from hop files
│       └── eda_inspect.py          # EDA script — hop counts, distributions
├── src/models/
│   └── baseline.py                 # DistilBERT fine-tuning (Stage 1)
├── checkpoints/
│   └── distilbert-baseline/        # Saved model (auto-generated after training)
│       ├── config.json
│       ├── pytorch_model.bin
│       └── results.json            # Final test metrics
├── requirements.txt
└── README.md
```

---

## Dataset

MetaQA (MoviE Text Audio QA) — a movie knowledge graph QA dataset with 1-hop, 2-hop, and 3-hop questions.

**Download:** https://drive.google.com/drive/folders/0B-36Uca2AvwhTWVFSUZqRXVtbUE

Download only the **vanilla text files** — no audio, no NTM needed:
- `kb.txt` from the root
- `1-hop/vanilla/` → rename folder to `1hop/`, move files up
- `2-hop/vanilla/` → rename to `2hop/`
- `3-hop/vanilla/` → rename to `3hop/`

**Dataset size:**

| Split | 1-hop | 2-hop | 3-hop | Total |
|-------|------:|------:|------:|------:|
| Train | 96,106 | 118,980 | 114,196 | 329,282 |
| Dev | 9,992 | 14,872 | 14,274 | 39,138 |
| Test | 9,947 | 14,872 | 14,274 | 39,093 |

**Knowledge base:** 134,741 triples | 43,234 unique entities | movie domain

---

## Setup

```bash
# 1. Clone and create virtual environment
git clone <repo-url>
cd pocket-graphrag
python -m venv .venv
.venv\Scripts\activate       # Windows
source .venv/bin/activate    # Linux/Mac

# 2. Install dependencies
pip install -r requirements.txt

# 3. Download MetaQA data (see above) and place in data/raw/

# 4. Login to WandB
wandb login
```

---

## Run EDA

```bash
python src/data/eda_inspect.py --data_dir data/raw --save_report
```

Prints hop counts, KB statistics, answer distributions, and sample QA pairs. Saves a JSON summary to `data/processed/eda_report.json`.

---

## Train DistilBERT Baseline

```bash
python src/models/baseline.py \
    --data_dir data/raw \
    --kb_path data/raw/kb.txt \
    --output_dir checkpoints/distilbert-baseline \
    --samples_per_hop 16667 \
    --epochs 6
```

**Key arguments:**

| Argument | Default | Description |
|----------|---------|-------------|
| `--samples_per_hop` | 16667 | Train examples per hop (16667×3 = 50k total) |
| `--epochs` | 6 | Training epochs |
| `--batch_size` | 16 | Batch size |
| `--grad_accum` | 2 | Gradient accumulation (effective batch = 32) |
| `--lr` | 3e-5 | Learning rate |
| `--no_cache` | False | Force dataset rebuild (ignores cache) |

**First run:** Dataset building takes ~3 hours (3-hop subgraph tokenization is slow). Results are cached to `data/processed/cache/` — every subsequent run loads in seconds.

**Hardware:** Tested on NVIDIA RTX 5070 8GB (PyTorch nightly + CUDA 12.8). Training takes ~7 min/epoch.

---

## How it works

MetaQA questions have a bracketed topic entity, e.g. `[Tom Hanks] appears in which movies`. The baseline:

1. Extracts the topic entity from the brackets
2. Builds a context by serialising the N-hop subgraph of that entity from `kb.txt` into plain text: `Cast Away starred_actors Tom Hanks. Philadelphia starred_actors Tom Hanks. ...`
3. Fine-tunes DistilBERT as an extractive QA model — given (question, context), predict the answer span
4. Uses **bidirectional adjacency** so traversal works in both directions (critical — many answers are reached by following reverse edges)
5. Uses **fp16 mixed precision** and **gradient accumulation** for memory efficiency on 8GB VRAM

---

## Stage 1 Baseline Results

Trained on 50,000 examples (16,667 per hop), 6 epochs, evaluated on 2,000 test samples per hop.

| Hop | Exact Match | F1 | Latency |
|-----|------------:|---:|--------:|
| 1-hop | 0.449 | 0.488 | 5.2ms |
| 2-hop | 0.649 | 0.674 | 6.1ms |
| 3-hop | 0.059 | 0.061 | 5.7ms |

**Best dev EM:** 0.404 (epoch 6)

**Key observation:** 3-hop performance is near zero because the 3-hop subgraph generates thousands of tokens, far exceeding DistilBERT's 512-token limit. The relevant context is truncated before the answer. This is the core problem that GraphRAG + knowledge distillation is designed to solve.

**WandB:** All runs logged to `pocket-graphrag` project.

---

## Tech stack (Stage 1)

| Component | Tool |
|-----------|------|
| Dataset | MetaQA (vanilla text) |
| Model | `distilbert-base-uncased` |
| Training | HuggingFace Transformers + PyTorch |
| Mixed precision | `torch.amp` fp16 |
| Experiment tracking | Weights & Biases |
| Caching | Python pickle |

---

## Next stage

See branch `post-baseline` for:
- FAISS semantic retrieval over KB triples
- N-hop graph traversal
- GPT-4o teacher generation
- Qwen2.5 student fine-tuning via LoRA
