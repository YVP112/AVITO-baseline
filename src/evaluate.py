from __future__ import annotations

from typing import Iterable

import pandas as pd


def _safe_div(numerator: float, denominator: float) -> float:
    if denominator == 0:
        return 0.0
    return numerator / denominator


def compute_micro_metrics(
    y_true: Iterable[list[int]],
    y_pred: Iterable[list[int]],
) -> dict[str, float]:
    true_positive = 0
    false_positive = 0
    false_negative = 0

    for true_ids, pred_ids in zip(y_true, y_pred):
        true_set = set(true_ids)
        pred_set = set(pred_ids)

        true_positive += len(true_set & pred_set)
        false_positive += len(pred_set - true_set)
        false_negative += len(true_set - pred_set)

    precision = _safe_div(true_positive, true_positive + false_positive)
    recall = _safe_div(true_positive, true_positive + false_negative)
    f1 = _safe_div(2 * precision * recall, precision + recall) if (precision + recall) else 0.0

    return {
        "precision_micro": precision,
        "recall_micro": recall,
        "f1_micro": f1,
    }


def compute_should_split_accuracy(
    y_true: Iterable[bool],
    y_pred: Iterable[bool],
) -> float:
    pairs = list(zip(y_true, y_pred))
    if not pairs:
        return 0.0
    correct = sum(int(true_value == pred_value) for true_value, pred_value in pairs)
    return correct / len(pairs)


def evaluate_predictions(df: pd.DataFrame) -> dict[str, float]:
    metrics = compute_micro_metrics(
        y_true=df["targetSplitMcIds"],
        y_pred=df["predSplitMcIds"],
    )
    metrics["should_split_accuracy"] = compute_should_split_accuracy(
        y_true=df["shouldSplit"],
        y_pred=df["predShouldSplit"],
    )
    return metrics
