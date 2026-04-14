"""
src/evaluation/metrics.py
Common metrics for Stage 2 and Stage 3 experiments.
"""


def normalize_answer(text: str) -> str:
    return " ".join(text.strip().lower().split())


def exact_match(prediction: str, gold_answers: list[str]) -> float:
    pred = normalize_answer(prediction)
    gold = {normalize_answer(answer) for answer in gold_answers}
    return float(pred in gold)


def f1_score(prediction: str, gold_answers: list[str]) -> float:
    pred_tokens = normalize_answer(prediction).split()
    if not pred_tokens:
        return 0.0

    best = 0.0
    pred_set = set(pred_tokens)
    for gold in gold_answers:
        gold_tokens = normalize_answer(gold).split()
        if not gold_tokens:
            continue

        common = pred_set & set(gold_tokens)
        if not common:
            continue

        precision = len(common) / len(pred_tokens)
        recall = len(common) / len(gold_tokens)
        score = 2 * precision * recall / (precision + recall)
        best = max(best, score)

    return best
