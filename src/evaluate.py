from __future__ import annotations

from typing import Iterable

import pandas as pd
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_score, recall_score
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


def compute_binary_split_metrics(
    y_true: Iterable[bool],
    y_pred: Iterable[bool],
) -> dict[str, float]:
    y_true_list = [bool(value) for value in y_true]
    y_pred_list = [bool(value) for value in y_pred]
    if not y_true_list:
        return {
            "should_split_accuracy": 0.0,
            "should_split_precision": 0.0,
            "should_split_recall": 0.0,
            "should_split_f1": 0.0,
        }

    return {
        "should_split_accuracy": accuracy_score(y_true_list, y_pred_list),
        "should_split_precision": precision_score(y_true_list, y_pred_list, zero_division=0),
        "should_split_recall": recall_score(y_true_list, y_pred_list, zero_division=0),
        "should_split_f1": f1_score(y_true_list, y_pred_list, zero_division=0),
    }


def compute_coverage_metrics(
    y_true: Iterable[list[int]],
    y_pred: Iterable[list[int]],
) -> dict[str, float]:
    true_labels = [set(labels) for labels in y_true]
    pred_labels = [set(labels) for labels in y_pred]
    total_true = sum(len(labels) for labels in true_labels)
    total_pred = sum(len(labels) for labels in pred_labels)
    covered_true = sum(len(true & pred) for true, pred in zip(true_labels, pred_labels))

    return {
        "target_label_count": float(total_true),
        "predicted_label_count": float(total_pred),
        "covered_true_labels": float(covered_true),
        "coverage_recall": (covered_true / total_true) if total_true else 0.0,
    }


def compute_should_split_error_breakdown(
    y_true: Iterable[bool],
    y_pred: Iterable[bool],
) -> dict[str, float]:
    y_true_list = [bool(value) for value in y_true]
    y_pred_list = [bool(value) for value in y_pred]
    if not y_true_list:
        return {"tn": 0.0, "fp": 0.0, "fn": 0.0, "tp": 0.0}

    tn, fp, fn, tp = confusion_matrix(y_true_list, y_pred_list, labels=[False, True]).ravel()
    return {"tn": float(tn), "fp": float(fp), "fn": float(fn), "tp": float(tp)}


def evaluate_predictions(df: pd.DataFrame) -> dict[str, float]:
    pred_split_mc_ids = df["predDrafts"].apply(extract_mc_ids_from_drafts)
    metrics = compute_micro_metrics(
        y_true=df["targetSplitMcIds"],
        y_pred=pred_split_mc_ids,
    )
    metrics.update(
        compute_binary_split_metrics(
            y_true=df["shouldSplit"],
            y_pred=df["predShouldSplit"],
        )
    )
    metrics.update(
        compute_coverage_metrics(
            y_true=df["targetSplitMcIds"],
            y_pred=pred_split_mc_ids,
        )
    )
    metrics.update(
        compute_should_split_error_breakdown(
            y_true=df["shouldSplit"],
            y_pred=df["predShouldSplit"],
        )
    )
    return metrics


def build_error_analysis_frame(df: pd.DataFrame) -> pd.DataFrame:
    pred_split_mc_ids = df["predDrafts"].apply(extract_mc_ids_from_drafts)
    error_df = df.copy()
    error_df["predSplitMcIds"] = pred_split_mc_ids
    error_df["splitFp"] = (~error_df["shouldSplit"]) & (error_df["predShouldSplit"])
    error_df["splitFn"] = (error_df["shouldSplit"]) & (~error_df["predShouldSplit"])
    error_df["missedMcIds"] = [
        sorted(set(true_labels) - set(pred_labels))
        for true_labels, pred_labels in zip(error_df["targetSplitMcIds"], error_df["predSplitMcIds"])
    ]
    error_df["extraMcIds"] = [
        sorted(set(pred_labels) - set(true_labels))
        for true_labels, pred_labels in zip(error_df["targetSplitMcIds"], error_df["predSplitMcIds"])
    ]
    return error_df
