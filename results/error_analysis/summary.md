# Error Source Analysis Summary

- Best available overall EM row: Qwen2.5-3B GraphRAG Gold (0.489).
- Use Qwen2.5-3B GraphRAG Gold as the best answer-accuracy model in the report.
- Use Qwen2.5-3B GraphRAG Hybrid for the evidence-trace demo, because it is trained with teacher traces.
- Same-size base vs trained signal: 3B Hybrid improves overall EM over 3B base by 0.211; sample sizes differ, so frame this as suggestive context-use evidence.
- Teacher trace quality: average evidence support is 0.856, with unsupported-evidence examples at 0.221 across hops.
- Saved prediction examples show partial-overlap errors most strongly for Qwen2.5-0.5B GraphRAG Hybrid (0.277 of saved examples).

## Visuals

- `model_em_by_hop.png`: compare all available models by hop.
- `f1_em_gap_by_hop.png`: shows where F1 is much higher than EM, indicating partial answer-set overlap.
- `retrieval_coverage_by_hop.png`: gold answer coverage in GraphRAG/RAG contexts.
- `teacher_trace_quality.png`: evidence support and unsupported-evidence rate.
- `answer_burden_by_hop.png`: hop-wise multi-answer burden.
- `answer_set_error_breakdown.png`: saved prediction error modes.

## Framing

Call this failure mode analysis or error source analysis, not absolute causal proof. Retrieval, evidence grounding, path selection, formatting, and supervision noise interact.