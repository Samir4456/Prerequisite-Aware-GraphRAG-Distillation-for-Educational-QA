# Instruction Set Strategy for Gold SFT vs Distillation

This document summarizes the project decision around Alpaca-format instruction data, gold-answer supervised fine-tuning, teacher distillation, evidence distillation, batching, parent model selection, expected training time, and budget.

## 1. Project Goal

The goal is to train a small instruction model to answer MetaQA questions using RAG and GraphRAG context.

The core experiment should answer:

- Does graph context improve performance over plain RAG?
- Does fine-tuning a small model improve over zero-shot RAG/GraphRAG?
- Does teacher distillation improve over training directly on dataset gold answers?
- Does the improvement change across 1-hop, 2-hop, and 3-hop questions?

The recommended instruction format is Alpaca-style JSON:

```json
[
  {
    "instruction": "Answer the question using the retrieved context and knowledge graph. Return only the answer entity or entities separated by |.",
    "input": "Knowledge Graph:\n...\n\nRetrieved Context:\n...\n\nQuestion: ...",
    "output": "..."
  }
]
```

## 2. Gold Answers vs Distillation

### Gold-Answer Supervised Fine-Tuning

Gold-answer SFT means the output comes directly from MetaQA labels.

```text
Input: question + RAG/GraphRAG context
Output: dataset gold answer
```

Example:

```json
{
  "instruction": "Answer the question using the graph and retrieved context.",
  "input": "Knowledge Graph:\nForrest Gump -> starred_actors -> Tom Hanks\nCast Away -> starred_actors -> Tom Hanks\n\nRetrieved Context:\n- Forrest Gump starred_actors Tom Hanks\n- Cast Away starred_actors Tom Hanks\n\nQuestion: What movies did Tom Hanks star in?",
  "output": "Forrest Gump | Cast Away"
}
```

This is not true teacher distillation. It is supervised fine-tuning on gold labels.

Use this when:

- You want the cheapest and most reliable training data.
- You want clean automatic evaluation with EM/F1.
- You want to avoid teacher hallucinations.
- You want a strong baseline before adding teacher-generated data.

### Answer-Only Teacher Distillation

Answer-only distillation means the output comes from a parent/teacher model, but the output is still only the final answer.

```text
Input: question + RAG/GraphRAG context
Output: teacher-generated final answer
```

This is distillation, but it is a weak form if the teacher answer is identical to the dataset gold answer.

Use this when:

- You want to test whether a stronger teacher cleans up noisy labels.
- You want the student to imitate answer formatting.
- You are using a dataset where gold answers are incomplete or inconsistent.

For MetaQA, answer-only distillation may not add much because the gold answers are already structured.

### Teacher Evidence Distillation

Evidence distillation means the teacher selects supporting graph triples or retrieved chunks, then provides the final answer.

```text
Input: question + RAG/GraphRAG context
Output: supporting evidence + final answer
```

Example:

```json
{
  "instruction": "Use the graph and retrieved context to answer the question. First list supporting evidence, then give the final answer.",
  "input": "Knowledge Graph:\nChristopher Nolan -> directed_by_inverse -> Inception\nInception -> starred_actors -> Leonardo DiCaprio\nLeonardo DiCaprio -> starred_actors_inverse -> Titanic\n\nRetrieved Context:\n- Inception starred_actors Leonardo DiCaprio\n- Titanic starred_actors Leonardo DiCaprio\n\nQuestion: Which movies star actors from movies directed by Christopher Nolan?",
  "output": "Supporting evidence:\n- Christopher Nolan is connected to Inception through directed_by_inverse.\n- Inception is connected to Leonardo DiCaprio through starred_actors.\n- Leonardo DiCaprio is connected to Titanic through starred_actors_inverse.\n\nFinal answer: Titanic"
}
```

This is a stronger distillation setup because the model is not only learning the answer string; it is also learning evidence selection.

Use this when:

- You want a real distillation claim.
- You want explainability in the demo.
- You want the model to learn graph evidence use.
- You want to study whether evidence traces help 2-hop and 3-hop questions.

### Hybrid Gold Answer + Teacher Evidence

This is the recommended research path.

```text
Input: question + RAG/GraphRAG context
Output: teacher-selected evidence + dataset gold final answer
```

Why this is strong:

- The final answer stays clean and evaluation-friendly.
- The teacher contributes a real distillation signal through evidence selection.
- The model learns how to use GraphRAG context.
- It reduces the risk of teacher hallucinated final answers.

Recommended output:

```text
Supporting evidence:
- ...
- ...

Final answer: ...
```

During evaluation, parse only the text after `Final answer:`.

## 3. Should We Include Reasoning?

Do not train the main evaluation model on free-form chain-of-thought.

Prefer structured evidence:

```text
Supporting evidence:
- triple or context sentence
- triple or context sentence

Final answer: answer1 | answer2
```

Reasons:

- It is easier to parse than free-form reasoning.
- It is safer than asking for hidden chain-of-thought.
- It makes the graph contribution visible.
- It supports both explainable demo output and clean EM/F1 evaluation.

For the main benchmark model, answer-only output is still useful:

```text
Final answer only:
answer1 | answer2
```

For the distillation model, structured evidence plus final answer is better.

## 4. Recommended Experiment Matrix

Build these datasets in order.

| Dataset | Input | Output | Purpose |
|---|---|---|---|
| Gold RAG SFT | Question + retrieved chunks | MetaQA gold answer | Plain RAG fine-tuning baseline |
| Gold GraphRAG SFT | Question + retrieved chunks + graph triples | MetaQA gold answer | Tests graph-augmented SFT |
| Teacher Answer Distillation | Question + context | Teacher final answer | Tests answer-only distillation |
| Teacher Evidence Distillation | Question + context | Teacher evidence + teacher final answer | Tests reasoning/evidence transfer |
| Hybrid Evidence + Gold Answer | Question + context | Teacher evidence + MetaQA gold answer | Best research-quality setup |

Recommended final comparison:

| System | Training Signal | Expected Use |
|---|---|---|
| DistilBERT baseline | Already trained | Stage 1 baseline |
| Zero-shot RAG | No fine-tuning | Stage 2 baseline |
| Zero-shot GraphRAG | No fine-tuning | Stage 3 baseline |
| Qwen2.5 Gold GraphRAG SFT | Gold answer | Strong practical student |
| Qwen2.5 Evidence-Distilled GraphRAG | Teacher evidence + gold answer | Main research student |

## 5. Parent Model Recommendation

Use an instruction-tuned parent model, not a base model.

Recommended student model order:

1. `Qwen/Qwen2.5-1.5B-Instruct`
2. `Qwen/Qwen2.5-3B-Instruct`
3. `Qwen/Qwen2.5-7B-Instruct`

Why start with 1.5B:

- It is small enough for fast LoRA experiments.
- It can generate answer strings, unlike extractive BERT.
- It is realistic for a "Pocket GraphRAG" local system.
- It is a good first point for model-size ablation.

Why test 3B:

- Better reasoning capacity than 1.5B.
- Still easier to train and serve than 7B.
- Useful midpoint for the professor's model-size question.

Why test 7B:

- Strong upper-bound student model.
- Better chance of learning multi-hop graph behavior.
- Good comparison against 1.5B and 3B.

Use the same instruction format for all sizes so the ablation is fair.

## 6. Teacher Options

### Option A: No Teacher, Gold Only

Use MetaQA gold answers directly.

Best for:

- Fast implementation.
- Low/no API budget.
- Clean labels.
- First working fine-tuning run.

This is not distillation. Call it gold-answer supervised fine-tuning.

### Option B: API Teacher

Use an API teacher to produce answers or evidence traces.

Best for:

- Strong teacher quality.
- Better distillation story.
- Minimal local setup.

Costs money, but for small datasets the cost is manageable.

### Option C: Local Teacher with vLLM

Use a larger local instruction model served by vLLM.

Best for:

- Batch generation without API cost.
- Fully local pipeline.
- Repeated experiments once set up.

Use vLLM when:

- You have enough GPU VRAM.
- You want to generate many teacher outputs.
- You want OpenAI-compatible local endpoints or offline batch inference.

Do not use vLLM when:

- You are using dataset gold answers.
- You are using an API teacher.
- You only need a small pilot set.

### Option D: Hybrid

Use teacher-selected evidence but dataset gold final answers.

This is the recommended distillation path for this project.

## 7. Batch Instruction Set Generation

Do not manually ask questions one at a time.

The correct workflow is:

1. Load all MetaQA questions for 1-hop, 2-hop, and 3-hop.
2. Randomly sample the desired number per hop.
3. For each question, run the RAG or GraphRAG pipeline.
4. Build the Alpaca `instruction`, `input`, and `output`.
5. Save everything to JSON.

### Batch Gold Instruction Set

This is the fastest path.

For each example:

```python
instruction = "Answer the question using the retrieved context and knowledge graph. Return only the answer entity or entities separated by |."
input = graph_text + retrieved_context + question
output = " | ".join(gold_answers)
```

No API calls are needed.

Target files:

```text
data/processed/instruction_pairs/train_rag_gold.json
data/processed/instruction_pairs/train_graphrag_gold.json
data/processed/instruction_pairs/train_mixed_gold.json
```

### Batch API Teacher Instruction Set

Use this for teacher answers or teacher evidence.

Workflow:

1. Build all prompts locally.
2. Save them as requests in a JSONL batch file.
3. Submit the JSONL file to the Batch API.
4. Wait for completion.
5. Download results.
6. Merge teacher outputs back into Alpaca JSON.

OpenAI Batch API is useful because it is asynchronous, has separate batch limits, and is priced at 50 percent lower cost than standard processing. The official Batch API guide says it is intended for jobs that do not require immediate responses and may complete within a 24-hour window. Source: https://platform.openai.com/docs/guides/batch

Batch request shape:

```jsonl
{"custom_id":"graphrag_1hop_000001","method":"POST","url":"/v1/responses","body":{"model":"gpt-4.1-mini","input":"You are creating training data...\n\nQuestion: ...\nContext: ..."}}
{"custom_id":"graphrag_1hop_000002","method":"POST","url":"/v1/responses","body":{"model":"gpt-4.1-mini","input":"You are creating training data...\n\nQuestion: ...\nContext: ..."}}
```

The `custom_id` should encode:

```text
mode_hop_index
```

Examples:

```text
rag_1hop_000001
graphrag_2hop_000148
graphrag_3hop_001500
```

This makes merging results deterministic.

### Batch Local Teacher with vLLM

Use this if you want local teacher generation.

Workflow:

1. Start a vLLM server for a larger teacher model.
2. Build prompts from all sampled examples.
3. Send prompts in parallel or use vLLM offline inference.
4. Save outputs.
5. Convert outputs to Alpaca JSON.

Recommended local teacher candidates:

- `Qwen/Qwen2.5-7B-Instruct`
- `Qwen/Qwen2.5-14B-Instruct` if hardware allows
- Another strong local instruct model available on your machine

Use local vLLM if API cost is a concern and you have GPU capacity.

## 8. Dataset Sizes

Recommended progression:

| Stage | Examples | Purpose |
|---|---:|---|
| Smoke test | 30 total, 10 per hop | Verify formatting and pipeline |
| Pilot | 300 total, 100 per hop | Check training and evaluation |
| Small run | 1,500 total, 500 per hop | Good first report result |
| Medium run | 6,000 total, 2,000 per hop | Stronger result if time allows |
| Full run | 15,000+ total | Only if training is stable |

For the report, a balanced 1,500-example or 6,000-example dataset is enough to show meaningful trends.

## 9. Budget Estimates

These are planning estimates, not guaranteed bills. Actual cost depends on prompt length, output length, selected model, retries, and whether Batch API is used.

Current official OpenAI pricing pages list standard token prices per 1M tokens and note that Batch API can reduce input and output costs by 50 percent. Source: https://openai.com/api/pricing and https://platform.openai.com/docs/guides/batch

Assumption for teacher evidence generation:

```text
Average input: 900 tokens per example
Average output: 180 tokens per example
Total: 1,080 tokens per example
```

Approximate standard API token volume:

| Examples | Input Tokens | Output Tokens |
|---:|---:|---:|
| 300 | 270K | 54K |
| 1,500 | 1.35M | 270K |
| 6,000 | 5.4M | 1.08M |
| 15,000 | 13.5M | 2.7M |

Approximate cost examples using commonly suitable teacher tiers:

| Teacher | Standard Price Pattern | 1,500 Examples | 6,000 Examples |
|---|---|---:|---:|
| `gpt-4.1-mini` | low-cost teacher | low single-digit USD | low tens USD |
| `gpt-4o-mini` | very cheap answer/evidence teacher | under 1 USD to a few USD | a few USD |
| `gpt-4o` or stronger | higher quality, higher cost | several USD | tens USD |

Batch API roughly halves those API costs if the chosen model supports Batch.

Suggested budget plan:

| Phase | Data | Method | Budget |
|---|---|---|---:|
| Gold SFT smoke test | 30 examples | no teacher | $0 |
| Gold SFT pilot | 300 to 1,500 examples | no teacher | $0 |
| Teacher evidence pilot | 300 examples | API Batch | usually under a few dollars |
| Teacher evidence run | 1,500 examples | API Batch | usually a few dollars |
| Ablations | 1,500 to 6,000 examples | API Batch/local teacher | reserve $10 to $30 |

For this project, reserve around:

```text
$0 if using gold-only SFT
$5 to $15 for teacher evidence at small scale
$15 to $30 for teacher evidence plus ablations
```

If using local vLLM teacher, API cost is $0, but compute time and GPU availability become the limiting factors.

## 10. Training Time Estimates

These are rough local LoRA estimates. Actual time depends on GPU, sequence length, batch size, quantization, gradient accumulation, and dataset size.

Assumptions:

- LoRA or QLoRA.
- Sequence length around 1,024 to 2,048 tokens.
- 1 to 3 epochs.
- Consumer GPU or cloud notebook GPU.

| Model | 1,500 Examples | 6,000 Examples | Notes |
|---|---:|---:|---|
| Qwen2.5-0.5B-Instruct | 10 to 30 min | 30 to 90 min | Fastest sanity check |
| Qwen2.5-1.5B-Instruct | 30 to 90 min | 1.5 to 4 hours | Recommended first real model |
| Qwen2.5-3B-Instruct | 1 to 3 hours | 3 to 8 hours | Good middle ablation |
| Qwen2.5-7B-Instruct | 3 to 8 hours | 8 to 20+ hours | Best upper-bound student |

If using CPU only, training is likely impractical for 3B and 7B. Use LoRA/QLoRA on GPU.

## 11. Implementation Steps

### Step 1: Add an Alpaca Exporter

Create a script:

```text
src/teacher/build_instruction_set.py
```

It should support:

```text
--mode rag|graphrag|mixed
--label_source gold|teacher_answer|teacher_evidence|hybrid
--samples_per_hop 500
--split train
--output_path data/processed/instruction_pairs/train_graphrag_gold.json
```

For each example:

1. Load QA pair.
2. Run RAG/GraphRAG context builder.
3. Build Alpaca object.
4. Save JSON.

### Step 2: Build Gold Datasets First

Create:

```text
train_rag_gold.json
train_graphrag_gold.json
train_mixed_gold.json
```

Start with:

```text
500 examples per hop = 1,500 total
```

This is enough for a first model.

### Step 3: Train Qwen2.5-1.5B-Instruct

Use LlamaFactory or Unsloth.

Recommended first training:

```text
model: Qwen/Qwen2.5-1.5B-Instruct
dataset: train_graphrag_gold.json
method: LoRA
epochs: 2 to 3
sequence length: 1024 or 2048
```

Evaluate separately on:

```text
1-hop test
2-hop test
3-hop test
```

### Step 4: Add Teacher Evidence Distillation

Once gold SFT works:

1. Generate teacher evidence for the same sampled examples.
2. Use gold answer as final answer.
3. Train another Qwen2.5-1.5B model.
4. Compare against Gold GraphRAG SFT.

### Step 5: Run Ablations

Recommended ablations:

- RAG vs GraphRAG input.
- Gold SFT vs teacher evidence distillation.
- 1.5B vs 3B vs 7B.
- K values: 3, 5, 10, 15.
- Hop depth: 1, 2, 3.

## 12. Recommended Final Path

The safest path:

```text
1. Build Gold GraphRAG Alpaca dataset.
2. Train Qwen2.5-1.5B-Instruct.
3. Evaluate on 1-hop, 2-hop, and 3-hop.
4. Build Hybrid Evidence + Gold Answer dataset.
5. Train same model again.
6. Compare Gold SFT vs Evidence Distillation.
7. If time allows, repeat with Qwen2.5-3B-Instruct.
```

The strongest paper claim:

```text
Gold SFT teaches the student to answer graph-augmented QA examples.
Teacher evidence distillation adds an evidence-selection signal.
This signal should help more on 2-hop and 3-hop questions than on 1-hop questions.
```

## 13. What to Call Each Method

Use precise naming in the report:

| Method | Correct Name |
|---|---|
| MetaQA answer only | Gold-answer supervised fine-tuning |
| Teacher answer only | Answer distillation |
| Teacher reasoning/evidence + teacher answer | Evidence/reasoning distillation |
| Teacher evidence + MetaQA answer | Hybrid evidence distillation |
| Graph context + answer training | Graph-augmented SFT |

Avoid claiming "distillation" if the teacher does not produce any part of the training target.

## 14. Final Recommendation

Start with gold-answer GraphRAG SFT because it is cheap, reliable, and easy to evaluate.

Then add hybrid evidence distillation because it provides a real teacher signal without corrupting the final answer labels.

Use vLLM only if you want local batch teacher generation and have enough GPU. Otherwise, use the OpenAI Batch API for teacher evidence generation, or skip teacher generation entirely for the first training run.

