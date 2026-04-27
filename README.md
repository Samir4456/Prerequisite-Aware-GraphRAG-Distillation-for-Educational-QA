# Pocket GraphRAG
### Small-Model GraphRAG Distillation for Multi-Hop Knowledge-Graph QA

> A controlled MetaQA benchmark study of **base Qwen**, **GraphRAG Gold supervision**, and **GraphRAG Hybrid trace supervision**, with a second layer of **error source analysis** to explain why performance changes across 1-hop, 2-hop, and 3-hop questions.

[![WandB](https://img.shields.io/badge/WandB-pocket--graphrag-yellow)](https://wandb.ai/st125989-asian-institute-of-technology/pocket-graphrag)
[![Python](https://img.shields.io/badge/Python-3.10-blue)](https://python.org)
[![License](https://img.shields.io/badge/License-MIT-green)](LICENSE)

---

## Overview

This project asks a focused question:

**Can small Qwen2.5 models answer multi-hop knowledge-graph questions better when we combine graph retrieval with targeted supervision?**

We evaluate on **MetaQA**, a movie-domain KGQA benchmark with:

- a prebuilt knowledge graph
- bracketed topic entities
- gold answers
- explicit 1-hop, 2-hop, and 3-hop splits

The project compares three main regimes:

- **Base Qwen GraphRAG**: no fine-tuning, only prompted with graph/text context
- **GraphRAG Gold**: fine-tuned on gold final answers
- **GraphRAG Hybrid**: fine-tuned on teacher-style evidence traces plus gold final answers

The key contribution is not only performance reporting. We also generate **failure mode / error source analysis** to explain likely causes such as:

- retrieval coverage failure
- unsupported evidence / grounding failure
- answer-set incompleteness
- teacher trace compression
- multi-answer burden on 3-hop questions

---

## Current Headline Results

### Best answer-accuracy model

- **Qwen2.5-3B GraphRAG Gold**
- `overall EM = 0.489`

### Best evidence-trace demo model

- **Qwen2.5-3B GraphRAG Hybrid**
- latest saved eval with trace prompting:
  - `1-hop EM = 0.824`, `F1 = 0.9088`
  - `2-hop EM = 0.556`, `F1 = 0.6986`
  - `3-hop EM = 0.042`, `F1 = 0.2381`
  - `overall EM = 0.474`, `overall F1 = 0.6152`

### Best framing for the report

- Use **3B Gold** as the strongest final-answer model.
- Use **3B Hybrid** as the live evidence-trace and error-analysis demo model.

---

## What The Results Mean

The raw benchmark metrics tell you **what happened**:

- EM
- F1
- latency

The error-analysis layer tells you **why it likely happened**:

- If gold answers are missing from context, the likely issue is **retrieval failure**.
- If evidence is unsupported, the likely issue is **grounding / hallucination failure**.
- If evidence is real but the chain is wrong, the likely issue is **path-selection / reasoning failure**.
- If F1 is decent but EM is low, the likely issue is often **answer-set incompleteness or overgeneration**, not total failure.

This is the reason the project is framed as:

- `failure mode analysis`
- `error source analysis`

and not as absolute causal proof.

---

## Main Findings From Error Analysis

From the latest saved `results/error_analysis/summary.md`:

- Best available overall EM row: **Qwen2.5-3B GraphRAG Gold (0.489)**
- 3B Hybrid improves overall EM over 3B base by **0.227**
- Teacher trace quality:
  - average evidence support: **0.856**
  - unsupported-evidence examples: **0.221**
- Teacher trace completeness:
  - grounded evidence covers every gold answer in only **0.446** of hybrid examples on average
  - average grounded compression gap: **6.01** gold answers
- Direct-answer artifact rate:
  - **0.372** of hybrid examples have a correct final answer but no grounded evidence line covering any gold answer

Interpretation:

- **3-hop is hard not only because reasoning is deeper**
- it is also hard because:
  - gold answer sets are larger
  - coverage drops
  - evidence traces become compressed
  - exact-match becomes unforgiving under multi-answer output burden

---

## Key Visuals

### Core benchmark visuals

Located in [`results/`](results):

- `all_em_by_hop.png`
- `all_f1_by_hop.png`
- `all_latency.png`
- `base_vs_finetuned.png`
- `improvement_delta.png`

### Error-analysis visuals

Located in [`results/error_analysis/`](results/error_analysis):

- `model_em_by_hop.png`
- `f1_em_gap_by_hop.png`
- `retrieval_coverage_by_hop.png`
- `answer_burden_by_hop.png`
- `teacher_trace_quality.png`
- `teacher_trace_gold_coverage.png`
- `teacher_trace_compression_gap.png`
- `answer_set_error_breakdown.png`

These are the main report / PPT-ready graphics.

---

## Demos

### 1. Live app

```powershell
streamlit run app.py
```

What it does:

- shows saved benchmark and error-analysis visuals immediately
- lets you run a live GraphRAG question through a selected local student checkpoint
- supports evidence-trace output for hybrid checkpoints

Recommended live choice:

- `Qwen2.5-1.5B GraphRAG Hybrid` for speed
- `Qwen2.5-3B GraphRAG Hybrid` for strongest trace-style demo if runtime is acceptable

### 2. Failure-mode analysis app

```powershell
streamlit run failure_mode_demo.py
```

What it does:

- all-model comparison
- teacher vs student trace comparison
- hop explorer
- saved prediction error exploration

---

## Quickstart

### 1. Environment

```bash
python -m venv .venv
source .venv/bin/activate
# Windows: .venv\Scripts\Activate.ps1
```

Install core packages:

```bash
pip install "numpy<2" faiss-cpu==1.7.4
pip install transformers datasets accelerate wandb peft
pip install sentence-transformers llamafactory streamlit pandas tqdm matplotlib pypdf
```

### 2. MetaQA data

Download MetaQA and place files as:

```text
data/raw/kb.txt
data/raw/1hop/qa_train.txt
data/raw/1hop/qa_dev.txt
data/raw/1hop/qa_test.txt
data/raw/2hop/...
data/raw/3hop/...
```

### 3. Build FAISS index

```bash
python src/retrieval/build_index.py --kb_path data/raw/kb.txt
```

### 4. Build instruction sets

Gold GraphRAG:

```bash
python src/teacher/build_instruction_set.py \
    --mode graphrag --label_source gold \
    --samples_per_hop 2000 --seed 42 \
    --output_path data/processed/instruction_pairs/train_graphrag_gold.json
```

Hybrid GraphRAG:

```bash
python src/teacher/build_instruction_set.py \
    --mode graphrag --label_source hybrid \
    --samples_per_hop 2000 --seed 42 \
    --teacher_provider deepseek \
    --teacher_model deepseek-chat \
    --output_path data/processed/instruction_pairs/train_graphrag_hybrid.json
```

### 5. Train / evaluate

Run the main sweep:

```bash
python run_all.py --experiments gold hybrid --sizes 0.5b 1.5b 3b
```

Or evaluate one checkpoint:

```bash
python src/evaluation/evaluate_student.py \
    --model_path checkpoints/qwen2.5-3b-graphrag-hybrid \
    --mode graphrag \
    --run_name qwen2.5-3b-graphrag-hybrid \
    --n_samples 500 \
    --trace_output
```

### 6. Build error-analysis outputs

```bash
python src/evaluation/error_analysis_report.py
```

Important:

- this **does not rerun the models**
- it reads saved `results/*` artifacts and regenerates `results/error_analysis/*`

---

## Evaluation Outputs

### Standard evaluation

Saved per model in:

```text
results/<run_name>/eval_results.json
results/<run_name>/eval_examples.json
```

Main metrics:

- Exact Match (EM)
- F1
- latency per hop

### Error-analysis outputs

Saved in:

```text
results/error_analysis/
```

Important files:

- `model_metrics_all.csv`
- `dataset_trace_summary_by_hop.csv`
- `answer_set_error_breakdown.csv`
- `saved_example_answer_errors.csv`
- `summary.md`

---

## Current Project Framing

Use this wording in the report / slides:

> Pocket GraphRAG is a controlled MetaQA benchmark study showing that GraphRAG helps small Qwen models on multi-hop KGQA, while a second error-analysis layer explains why failures remain, especially on 3-hop questions.

And for the conclusion:

> 3-hop errors are driven not only by deeper reasoning difficulty, but also by lower retrieval coverage, larger answer sets, and compressed or incomplete evidence supervision.

---

## Repository Structure

```text
app.py
failure_mode_demo.py
run_all.py
results_charts.ipynb
configs/
data/
checkpoints/
results/
src/
  data/
  graph/
  retrieval/
  teacher/
  evaluation/
```

---

## Troubleshooting

| Error | Likely cause | Fix |
|---|---|---|
| `ValueError: We need an offload_dir...` | Model too large for available VRAM | use the patched app loaders with disk offload; prefer 1.5B Hybrid for live demo |
| `TypeError: image() got unexpected keyword` | older/newer Streamlit API mismatch | restart after latest `app.py` patch |
| `faiss import error` | numpy 2.x incompatibility | `pip install "numpy<2" faiss-cpu==1.7.4` |
| `CUDA out of memory` | large checkpoint or long trace generation | use 0.5B/1.5B model, reduce trace use, or allow CPU/disk offload |
| `checkpoint not found` | nested checkpoint folder layout | current app supports both `checkpoints/<model>` and `checkpoints/checkpoints/<model>` |

---

## Team

Asian Institute of Technology

WandB:
[pocket-graphrag](https://wandb.ai/st125989-asian-institute-of-technology/pocket-graphrag)

GitHub:
[Samir4456/Prerequisite-Aware-GraphRAG-Distillation-for-Educational-QA](https://github.com/Samir4456/Prerequisite-Aware-GraphRAG-Distillation-for-Educational-QA)
