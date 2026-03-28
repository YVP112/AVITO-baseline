from __future__ import annotations

from typing import Iterable

import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.preprocessing import MultiLabelBinarizer


def extract_mc_ids_from_drafts(drafts: object) -> list[int]:
    if not isinstance(drafts, list):
        return []

    mc_ids: list[int] = []
    for draft in drafts:
        if isinstance(draft, dict) and "mcId" in draft:
            mc_ids.append(int(draft["mcId"]))
    return mc_ids


def _to_multilabel_arrays(
    y_true: Iterable[list[int]],
    y_pred: Iterable[list[int]],
) -> tuple:
    true_labels = [list(labels) for labels in y_true]
    pred_labels = [list(labels) for labels in y_pred]

    all_labels = sorted({label for labels in true_labels + pred_labels for label in labels})
    mlb = MultiLabelBinarizer(classes=all_labels)

    y_true_binary = mlb.fit_transform(true_labels)
    y_pred_binary = mlb.transform(pred_labels)
    return y_true_binary, y_pred_binary


def compute_micro_metrics(
    y_true: Iterable[list[int]],
    y_pred: Iterable[list[int]],
) -> dict[str, float]:
    y_true_binary, y_pred_binary = _to_multilabel_arrays(y_true, y_pred)
    if y_true_binary.shape[1] == 0:
        return {
            "precision_micro": 0.0,
            "recall_micro": 0.0,
            "f1_micro": 0.0,
        }

    precision = precision_score(
        y_true_binary,
        y_pred_binary,
        average="micro",
        zero_division=0,
    )
    recall = recall_score(
        y_true_binary,
        y_pred_binary,
        average="micro",
        zero_division=0,
    )
    f1 = f1_score(
        y_true_binary,
        y_pred_binary,
        average="micro",
        zero_division=0,
    )

    return {
        "precision_micro": precision,
        "recall_micro": recall,
        "f1_micro": f1,
    }


def compute_should_split_accuracy(
    y_true: Iterable[bool],
    y_pred: Iterable[bool],
) -> float:
    y_true_list = list(y_true)
    y_pred_list = list(y_pred)
    if not y_true_list:
        return 0.0
    return accuracy_score(y_true_list, y_pred_list)


def evaluate_predictions(df: pd.DataFrame) -> dict[str, float]:
    pred_split_mc_ids = df["predDrafts"].apply(extract_mc_ids_from_drafts)
    metrics = compute_micro_metrics(
        y_true=df["targetSplitMcIds"],
        y_pred=pred_split_mc_ids,
    )
    metrics["should_split_accuracy"] = compute_should_split_accuracy(
        y_true=df["shouldSplit"],
        y_pred=df["predShouldSplit"],
    )
    return metrics
