# Error Analysis and Demo Guide

This project should present two different artifacts:

- **Report/PPT analysis:** aggregate failure-mode metrics and charts.
- **Demo:** an interactive failure inspector that shows one example at a time.

## Generate Report Outputs

Run:

```powershell
python src/evaluation/error_analysis_report.py
```

Outputs are written to:

```text
results/error_analysis/
```

Key files:

- `model_metrics_all.csv` compares all available reported/evaluated models.
- `dataset_trace_summary_by_hop.csv` contains retrieval/context coverage, answer-count burden, and teacher-trace quality by hop.
- `answer_set_error_breakdown.csv` summarizes incomplete-answer, overgeneration, exact-miss, and partial-overlap errors from saved predictions.
- `summary.md` gives short report-ready interpretation.
- `model_em_by_hop.png`, `retrieval_coverage_by_hop.png`, `teacher_trace_quality.png`, `answer_burden_by_hop.png`, and `answer_set_error_breakdown.png` are PPT-ready figures.

## Recommended Demo

Run:

```powershell
streamlit run failure_mode_demo.py
```

Use the demo to show:

- all-model aggregate comparison;
- one evidence trace where the answer is grounded;
- the same example transformed into retrieval miss, hallucinated evidence, wrong supported path, and answer-set mismatch scenarios;
- saved model errors where F1 is non-zero but EM is zero.

This is more appropriate than only showing "the model works", because the thesis contribution is not just answer generation. The stronger demo is: **the system can inspect a failure and explain the likely error source.**

## Future Full Per-Model Error Analysis

The committed repository does not include `checkpoints/`, `data/raw/`, or `data/faiss/`, so the current analysis uses committed result and instruction artifacts.

When those runtime artifacts are present, re-run evaluation with context saving:

```powershell
python run_all.py --eval_only --reeval --examples_limit 0 --save_context
python src/evaluation/error_analysis_report.py
```

That will allow per-model calculation of:

- retrieval coverage;
- evidence validity;
- path-faithfulness labels;
- partial overlap vs exact miss;
- answer-set incompleteness and overgeneration;
- same-size base vs trained deltas.

## Model Framing

Use this wording:

```text
Qwen2.5-3B GraphRAG Gold is the best overall answer-accuracy model.
Qwen2.5-3B GraphRAG Hybrid is the best evidence-trace demo model because it was trained with teacher evidence traces.
```

Call the method **failure mode analysis** or **error source analysis**, not absolute causal proof.
