"""
Failure-mode utilities for GraphRAG QA examples.

The functions in this module are intentionally lightweight so they can be
used both by Streamlit demos and offline evaluation scripts.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Iterable


FINAL_ANSWER_RE = re.compile(r"(?is)final\s+answer\s*:\s*(.+)$")
SUPPORTING_EVIDENCE_RE = re.compile(r"(?i)^\s*supporting\s+evidence\s*:?\s*$")


@dataclass(frozen=True)
class AnswerMetrics:
    exact_match: float
    f1: float
    precision: float
    recall: float
    missing: list[str]
    extra: list[str]


def normalize_text(value: str) -> str:
    """Normalize text for fuzzy-enough evidence and answer comparisons."""
    value = value.replace("\u2192", "->")
    value = value.replace("â†’", "->")
    value = re.sub(r"\s+", " ", value)
    return value.strip().lower()


def normalize_answer(value: str) -> str:
    """Normalize an answer entity while preserving meaningful punctuation."""
    value = re.sub(r"^(answer|final answer)\s*:\s*", "", value.strip(), flags=re.I)
    return normalize_text(value)


def dedupe_preserve_order(values: Iterable[str]) -> list[str]:
    seen = set()
    result = []
    for value in values:
        key = normalize_answer(value)
        if not key or key in seen:
            continue
        seen.add(key)
        result.append(value.strip())
    return result


def extract_final_answer(raw_output: str) -> str:
    """Return the text after 'Final answer:' when present, otherwise raw text."""
    match = FINAL_ANSWER_RE.search(raw_output or "")
    if match:
        return match.group(1).strip()
    return (raw_output or "").strip()


def parse_answer_list(raw_output: str | list[str]) -> list[str]:
    """Parse pipe-separated model output into answer entities."""
    if isinstance(raw_output, list):
        return dedupe_preserve_order(str(item) for item in raw_output)

    answer_text = extract_final_answer(raw_output)
    answer_text = re.sub(r"(?is)supporting\s+evidence\s*:.*", "", answer_text).strip()
    answer_text = re.sub(r"(?i)^final\s+answer\s*:\s*", "", answer_text).strip()
    if not answer_text:
        return []

    parts = [part.strip() for part in answer_text.split("|")]
    return dedupe_preserve_order(part for part in parts if part)


def clean_evidence_line(line: str) -> str:
    line = line.strip()
    line = re.sub(r"^\s*[-*]\s*", "", line)
    line = re.sub(r"^\s*\d+[.)]\s*", "", line)
    return line.strip()


def extract_evidence_lines(raw_output: str) -> list[str]:
    """Extract evidence bullets from a teacher or model trace."""
    lines = (raw_output or "").splitlines()
    evidence = []
    inside = False

    for line in lines:
        if SUPPORTING_EVIDENCE_RE.match(line):
            inside = True
            continue
        if re.match(r"(?i)^\s*final\s+answer\s*:", line):
            break
        if inside:
            cleaned = clean_evidence_line(line)
            if cleaned:
                evidence.append(cleaned)

    if evidence:
        return evidence

    # Fallback for traces that omit the header but still use bullets.
    for line in lines:
        if re.match(r"(?i)^\s*final\s+answer\s*:", line):
            break
        if re.match(r"^\s*[-*]\s+", line):
            evidence.append(clean_evidence_line(line))

    return evidence


def parse_context_sections(input_text: str) -> dict[str, list[str] | str]:
    """Split an Alpaca-style GraphRAG input into graph, retrieval, and question."""
    text = input_text or ""
    graph_text = ""
    retrieved_text = ""
    question = ""

    graph_match = re.search(
        r"(?is)Knowledge\s+Graph:\s*(.*?)(?:\n\s*\nRetrieved\s+Context:|\n\s*\nQuestion:|$)",
        text,
    )
    if graph_match:
        graph_text = graph_match.group(1).strip()

    retrieved_match = re.search(
        r"(?is)Retrieved\s+Context:\s*(.*?)(?:\n\s*\nQuestion:|$)",
        text,
    )
    if retrieved_match:
        retrieved_text = retrieved_match.group(1).strip()

    question_match = re.search(r"(?is)Question:\s*(.*)$", text)
    if question_match:
        question = question_match.group(1).strip()

    graph_lines = [clean_evidence_line(line) for line in graph_text.splitlines()]
    retrieved_lines = [clean_evidence_line(line) for line in retrieved_text.splitlines()]

    return {
        "graph_lines": [line for line in graph_lines if line],
        "retrieved_lines": [line for line in retrieved_lines if line],
        "question": question,
    }


def answer_metrics(predicted: list[str], gold: list[str]) -> AnswerMetrics:
    pred_map = {normalize_answer(value): value for value in predicted if normalize_answer(value)}
    gold_map = {normalize_answer(value): value for value in gold if normalize_answer(value)}

    pred_set = set(pred_map)
    gold_set = set(gold_map)
    common = pred_set & gold_set

    exact_match = float(pred_set == gold_set)
    precision = len(common) / len(pred_set) if pred_set else 0.0
    recall = len(common) / len(gold_set) if gold_set else 0.0
    f1 = 0.0
    if precision + recall:
        f1 = 2 * precision * recall / (precision + recall)

    missing = [gold_map[key] for key in sorted(gold_set - pred_set)]
    extra = [pred_map[key] for key in sorted(pred_set - gold_set)]

    return AnswerMetrics(
        exact_match=exact_match,
        f1=f1,
        precision=precision,
        recall=recall,
        missing=missing,
        extra=extra,
    )


def line_is_supported(evidence_line: str, context_lines: list[str]) -> bool:
    """Return True when an evidence line is grounded in retrieved/graph context."""
    evidence_norm = normalize_text(clean_evidence_line(evidence_line))
    if not evidence_norm:
        return False

    for context_line in context_lines:
        context_norm = normalize_text(clean_evidence_line(context_line))
        if not context_norm:
            continue
        if evidence_norm in context_norm or context_norm in evidence_norm:
            return True
    return False


def evidence_support(evidence_lines: list[str], context_lines: list[str]) -> dict:
    supported = []
    unsupported = []
    for line in evidence_lines:
        if line_is_supported(line, context_lines):
            supported.append(line)
        else:
            unsupported.append(line)

    total = len(evidence_lines)
    ratio = len(supported) / total if total else None
    return {
        "supported": supported,
        "unsupported": unsupported,
        "support_ratio": ratio,
        "total": total,
    }


def answer_context_coverage(answers: list[str], context_lines: list[str]) -> dict:
    joined_context = "\n".join(normalize_text(line) for line in context_lines)
    covered = []
    missing = []
    for answer in answers:
        if normalize_answer(answer) in joined_context:
            covered.append(answer)
        else:
            missing.append(answer)
    return {
        "covered": covered,
        "missing": missing,
        "all": bool(answers) and not missing,
        "any": bool(covered),
    }


def classify_failure(
    metrics: AnswerMetrics,
    gold_coverage: dict,
    pred_coverage: dict,
    support: dict,
    predicted: list[str],
) -> tuple[str, str]:
    """Assign a likely failure-mode label and a short explanation."""
    has_evidence = support["total"] > 0
    has_unsupported_evidence = bool(support["unsupported"])

    if metrics.exact_match == 1.0:
        if has_unsupported_evidence:
            return (
                "correct answer, unsupported evidence",
                "The final answer matches gold, but at least one evidence line is not grounded in the retrieved context.",
            )
        return (
            "correct grounded answer",
            "The predicted answer set matches gold, and no unsupported evidence was detected.",
        )

    if not gold_coverage["any"]:
        return (
            "retrieval miss",
            "None of the gold answers appear in the graph or retrieved context, so generation is not the first suspect.",
        )

    if has_unsupported_evidence:
        return (
            "hallucination / grounding failure",
            "The model cited evidence that is not present in the graph or retrieved context.",
        )

    if metrics.f1 > 0:
        if metrics.missing and metrics.extra:
            return (
                "partial overlap: incomplete answer set + extra answers",
                "The model found some gold answers but also missed required answers and added unsupported extras.",
            )
        if metrics.missing:
            return (
                "partial overlap: incomplete answer set",
                "F1 is non-zero but EM is zero because the model omitted at least one gold answer.",
            )
        if metrics.extra:
            return (
                "partial overlap: overgeneration",
                "The model included all gold answers but added extra wrong answers, so exact match fails.",
            )
        return (
            "formatting / set mismatch",
            "The answer overlaps gold, but the normalized sets still do not match exactly.",
        )

    if not predicted:
        return (
            "empty generation",
            "The model did not produce a parseable answer entity.",
        )

    if has_evidence and support["support_ratio"] == 1.0:
        return (
            "reasoning / path selection failure",
            "The evidence is grounded, but it does not lead to the gold answer set.",
        )

    if pred_coverage["any"]:
        return (
            "wrong relation or intermediate entity",
            "The predicted entity appears in context, but it is not the entity required by the question.",
        )

    return (
        "answer hallucination / grounding failure",
        "The predicted entity does not match gold and is not clearly supported by the context.",
    )


def analyze_case(
    gold_answers: list[str],
    raw_output: str = "",
    predicted_answers: list[str] | None = None,
    context_lines: list[str] | None = None,
    evidence_lines: list[str] | None = None,
) -> dict:
    """Analyze one QA example and return metrics plus likely error source."""
    context_lines = context_lines or []
    predicted = predicted_answers if predicted_answers is not None else parse_answer_list(raw_output)
    evidence = evidence_lines if evidence_lines is not None else extract_evidence_lines(raw_output)

    metrics = answer_metrics(predicted, gold_answers)
    support = evidence_support(evidence, context_lines)
    gold_coverage = answer_context_coverage(gold_answers, context_lines)
    pred_coverage = answer_context_coverage(predicted, context_lines)
    label, explanation = classify_failure(
        metrics=metrics,
        gold_coverage=gold_coverage,
        pred_coverage=pred_coverage,
        support=support,
        predicted=predicted,
    )

    return {
        "predicted": predicted,
        "gold": gold_answers,
        "evidence": evidence,
        "metrics": {
            "em": metrics.exact_match,
            "f1": metrics.f1,
            "precision": metrics.precision,
            "recall": metrics.recall,
            "missing": metrics.missing,
            "extra": metrics.extra,
        },
        "coverage": {
            "gold": gold_coverage,
            "predicted": pred_coverage,
        },
        "evidence_support": support,
        "failure_mode": label,
        "explanation": explanation,
    }
