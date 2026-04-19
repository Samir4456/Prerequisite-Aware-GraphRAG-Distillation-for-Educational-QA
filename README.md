# Pocket GraphRAG
### Prerequisite-Aware GraphRAG Distillation for Educational QA

> Fine-tuning small Qwen2.5 models (0.5B–3B) to perform multi-hop knowledge graph QA using GraphRAG, evaluated on MetaQA across 1-hop, 2-hop, and 3-hop reasoning tasks.

---

## Overview

This project investigates whether **small language models (SLMs)** can match large teacher models on multi-hop knowledge graph question answering when equipped with **graph-augmented retrieval (GraphRAG)** and **instruction fine-tuning**.

We compare:
- **RAG only** — flat FAISS retrieval over serialized KB triples
- **GraphRAG** — explicit KG traversal + FAISS retrieval combined
- **Gold SFT** — fine-tuned on MetaQA gold answers
- **Hybrid distillation** — fine-tuned on teacher-generated evidence chains + gold answers

Across three student model sizes: **Qwen2.5-0.5B**, **Qwen2.5-1.5B**, **Qwen2.5-3B**

---

## Key Findings

### GraphRAG vs RAG — 2-hop is the headline result

| Model | 1-hop EM | 2-hop EM | 3-hop EM | Overall EM |
|-------|---------|---------|---------|-----------|
| DistilBERT baseline | 0.449 | 0.649 | 0.059 | — |
| Qwen2.5-1.5B RAG only | 0.720 | 0.025 | 0.020 | 0.255 |
| Qwen2.5-1.5B GraphRAG Gold | 0.778 | 0.544 | 0.054 | 0.459 |
| Qwen2.5-3B GraphRAG Gold | **0.832** | **0.586** | 0.050 | **0.489** |
| Qwen2.5-3B GraphRAG Hybrid | 0.830 | 0.474 | **0.070** | 0.458 |

**RAG collapses on 2-hop (EM = 0.025)**. Without the knowledge graph, the model cannot connect entities across reasoning steps. GraphRAG maintains 0.544 on 2-hop — a **21x improvement** over RAG only.

### Model Size Scaling (GraphRAG Gold)

| Size | 1-hop EM | 2-hop EM | 3-hop EM | Overall EM |
|------|---------|---------|---------|-----------|
| 0.5B | 0.756 | 0.478 | 0.056 | 0.430 |
| 1.5B | 0.778 | 0.544 | 0.054 | 0.459 |
| 3B   | 0.832 | 0.586 | 0.050 | 0.489 |

### Fine-tuning vs Base Model

| Size | Type | 1-hop EM | 2-hop EM | 3-hop EM |
|------|------|---------|---------|---------|
| 0.5B | Base | 0.000 | 0.040 | 0.020 |
| 0.5B | Gold SFT | 0.756 | 0.478 | 0.056 |
| 1.5B | Base | 0.175 | 0.195 | 0.040 |
| 1.5B | Gold SFT | 0.778 | 0.544 | 0.054 |
| 3B | Base | 0.375 | 0.340 | 0.025 |
| 3B | Gold SFT | 0.832 | 0.586 | 0.050 |

---

## Pipeline

```
Question with [bracketed entity]
        │
        ▼
Entity Extraction
        │
   ┌────┴────┐
   ▼         ▼
KG Traversal  FAISS Retrieval
(N-hop)      (top-K chunks)
   │         │
   └────┬────┘
        ▼
  Context Prompt
  (graph triples + retrieved text)
        │
        ▼
  Qwen2.5 Student
        │
        ▼
    Answer
```

---

## Dataset

**MetaQA** — Movie knowledge graph QA benchmark

| Split | 1-hop | 2-hop | 3-hop |
|-------|-------|-------|-------|
| Train | 96,106 | 118,980 | 114,196 |
| Dev | 9,992 | 14,872 | 14,274 |
| Test | 9,947 | 14,872 | 14,274 |

- KB: **134,741 triples** | **43,234 entities** | **9 relation types**
- Questions contain bracketed topic entities: `who directed [Inception]`
- Answers are pipe-separated: `Christopher Nolan`

---

## Repository Structure

```
pocket-graphrag/
├── app.py                              # Streamlit demo
├── run_all.py                          # Master training + evaluation script
├── results_charts.ipynb               # Jupyter notebook for charts
├── dataset_info.json                   # LlamaFactory dataset registry
├── requirements.txt
├── configs/
│   ├── lora_0.5b_graphrag_gold.yaml
│   ├── lora_0.5b_graphrag_hybrid.yaml
│   ├── lora_1.5b_graphrag_gold.yaml
│   ├── lora_1.5b_graphrag_hybrid.yaml
│   ├── lora_1.5b_rag_gold.yaml
│   ├── lora_3b_graphrag_gold.yaml
│   └── lora_3b_graphrag_hybrid.yaml
├── data/
│   ├── raw/                            # MetaQA (download separately)
│   ├── faiss/                          # Auto-generated
│   └── processed/instruction_pairs/   # Generated JSON datasets
├── checkpoints/                        # Saved model weights
├── results/                            # Evaluation JSON + charts
└── src/
    ├── data/load_kb.py  load_metaqa.py  eda_inspect.py
    ├── graph/entity_extract.py  subgraph.py  serialize.py
    ├── retrieval/embedder.py  faiss_index.py  build_index.py
    ├── models/baseline.py
    ├── teacher/build_instruction_set.py
    └── evaluation/evaluate_student.py  compile_results.py
```

---

## Quickstart

### 1. Environment setup

```bash
python -m venv .venv
source .venv/bin/activate        # Linux/Mac
# .venv\Scripts\Activate.ps1    # Windows

pip install "numpy<2" faiss-cpu==1.7.4
pip install transformers==4.35.0 datasets accelerate wandb peft
pip install sentence-transformers llamafactory openai streamlit pandas tqdm matplotlib
```

**PyTorch — choose for your GPU:**
```bash
# RTX 5070/5080/5090 (Blackwell sm_120) — nightly required
pip install --pre torch torchvision torchaudio --index-url https://download.pytorch.org/whl/nightly/cu128

# RTX 3090/4090/A6000
pip install torch==2.4.1 --index-url https://download.pytorch.org/whl/cu124
```

### 2. Download MetaQA

```
https://drive.google.com/drive/folders/0B-36Uca2AvwhTWVFSUZqRXVtbUE
```

Arrange as:
```
data/raw/kb.txt
data/raw/1hop/qa_train.txt  qa_dev.txt  qa_test.txt
data/raw/2hop/  (same)
data/raw/3hop/  (same)
```

> Rename folders from `1-hop` to `1hop` after download.

### 3. Build FAISS index (once)

```bash
python src/retrieval/build_index.py --kb_path data/raw/kb.txt
```

### 4. Build instruction datasets

```bash
# Gold GraphRAG — free, no API
python src/teacher/build_instruction_set.py \
    --mode graphrag --label_source gold \
    --samples_per_hop 2000 --seed 42 \
    --output_path data/processed/instruction_pairs/train_graphrag_gold.json

# Hybrid teacher evidence — DeepSeek (~$0.05 per 1500 examples)
export DEEPSEEK_API_KEY="your-key"
python src/teacher/build_instruction_set.py \
    --mode graphrag --label_source hybrid \
    --samples_per_hop 2000 --seed 42 \
    --teacher_provider deepseek --teacher_model deepseek-chat \
    --output_path data/processed/instruction_pairs/train_graphrag_hybrid.json
```

### 5. Train all models overnight

```bash
python run_all.py --experiments gold hybrid --sizes 0.5b 1.5b 3b
```

### 6. Evaluate

```bash
python src/evaluation/evaluate_student.py \
    --model_path checkpoints/qwen2.5-3b-graphrag-gold \
    --mode graphrag --run_name qwen2.5-3b-graphrag-gold --n_samples 500
```

### 7. Compile results

```bash
python src/evaluation/compile_results.py --save_csv results/comparison.csv
```

### 8. Run demo

```bash
streamlit run app.py
```

---

## Training Configuration

| Parameter | 0.5B | 1.5B | 3B |
|-----------|------|------|----|
| `lora_rank` | 16 | 16 | 16 |
| `lora_alpha` | 32 | 32 | 32 |
| `batch_size` | 2 | 1 | 1 |
| `grad_accum` | 8 | 16 | 16 |
| `cutoff_len` | 512 | 512 | 512 |
| `epochs` | 3 | 3 | 3 |
| `lr` | 2e-4 | 2e-4 | 2e-4 |

---

## Instruction Set Format

**Gold label** (free, MetaQA gold answer):
```json
{
  "instruction": "Answer the question using the retrieved context and knowledge graph. Return only the answer entity or entities separated by |.",
  "input": "Knowledge Graph:\nInception -> directed_by -> Christopher Nolan\n\nRetrieved Context:\n- Inception directed_by Christopher Nolan\n\nQuestion: who directed Inception",
  "output": "Christopher Nolan"
}
```

**Hybrid label** (teacher evidence + gold answer):
```json
{
  "instruction": "Answer the question using the retrieved context and knowledge graph. First list the supporting evidence, then give the final answer.",
  "input": "Knowledge Graph:\nInception -> directed_by -> Christopher Nolan\n\nQuestion: who directed Inception",
  "output": "Supporting evidence:\n- Inception -> directed_by -> Christopher Nolan\n\nFinal answer: Christopher Nolan"
}
```

---

## Reproducibility

- All experiments use `--seed 42`
- Same seed + same MetaQA files = identical instruction pairs for all team members
- Teacher API outputs are not deterministic — generate once, share the JSON file

---

## WandB

All training and evaluation runs logged to:
```
https://wandb.ai/st125989-asian-institute-of-technology/pocket-graphrag
```

---

## Troubleshooting

| Error | Fix |
|-------|-----|
| `module 'inspect' has no attribute` | Rename `src/data/inspect.py` → `eda_inspect.py` |
| `faiss import error` | `pip install "numpy<2" faiss-cpu==1.7.4` |
| `CUDA out of memory` | Set `cutoff_len: 384` in yaml |
| `dataset_info.json not found` | Copy to `data/processed/instruction_pairs/` |
| `UnicodeEncodeError: charmap` | Add `encoding='utf-8'` to all `open()` calls |
| RTX 5070 not compatible | Use PyTorch nightly cu128 |

---

## Team

Asian Institute of Technology
WandB: [pocket-graphrag](https://wandb.ai/st125989-asian-institute-of-technology/pocket-graphrag)
GitHub: [Samir4456/Prerequisite-Aware-GraphRAG-Distillation-for-Educational-QA](https://github.com/Samir4456/Prerequisite-Aware-GraphRAG-Distillation-for-Educational-QA)