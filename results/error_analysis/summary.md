# Error Source Analysis Summary

- Best available overall EM row: Qwen2.5-3B GraphRAG Gold (0.489).
- Use Qwen2.5-3B GraphRAG Gold as the best answer-accuracy model in the report.
- Use Qwen2.5-3B GraphRAG Hybrid for the evidence-trace demo, because it is trained with teacher traces.
- Same-size base vs trained signal: 3B Hybrid improves overall EM over 3B base by 0.227; sample sizes differ, so frame this as suggestive context-use evidence.
- Teacher trace quality: average evidence support is 0.856, with unsupported-evidence examples at 0.221 across hops.
- Teacher trace completeness: grounded evidence covers every gold answer in 0.446 of hybrid examples on average; the average grounded compression gap is 6.01 gold answers.
- Direct-answer supervision artifact rate: 0.372 of hybrid examples have a correct final answer but no grounded evidence line covering any gold answer.
- Saved prediction examples show partial-overlap errors most strongly for Qwen2.5-0.5B GraphRAG Hybrid (0.390 of saved examples).

## Visuals

- `model_em_by_hop.png`: compare all available models by hop.
- `f1_em_gap_by_hop.png`: shows where F1 is much higher than EM, indicating partial answer-set overlap.
- `retrieval_coverage_by_hop.png`: gold answer coverage in GraphRAG/RAG contexts.
- `teacher_trace_quality.png`: evidence support and unsupported-evidence rate.
- `teacher_trace_gold_coverage.png`: whether teacher evidence covers gold answers.
- `teacher_trace_compression_gap.png`: evidence compression under multi-answer burden.
- `answer_burden_by_hop.png`: hop-wise multi-answer burden.
- `answer_set_error_breakdown.png`: saved prediction error modes.

## Framing

Call this failure mode analysis or error source analysis, not absolute causal proof. Retrieval, evidence grounding, path selection, formatting, and supervision noise interact.