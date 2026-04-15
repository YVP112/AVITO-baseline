from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from src.contracts import PredictionResult


def ensure_parent_dir(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def save_dataframe_csv(df: pd.DataFrame, path: Path) -> None:
    ensure_parent_dir(path)
    df.to_csv(path, index=False, encoding="utf-8-sig")


def save_prediction_results_jsonl(results: list[PredictionResult], path: Path) -> None:
    ensure_parent_dir(path)
    with path.open("w", encoding="utf-8") as file:
        for result in results:
            record = {
                "itemId": result.itemId,
                "detectedMcIds": result.detectedMcIds,
                "shouldSplit": result.shouldSplit,
                "drafts": [draft.__dict__ for draft in result.drafts],
                "categoryDecisions": [decision.__dict__ for decision in result.categoryDecisions],
            }
            file.write(json.dumps(record, ensure_ascii=False) + "\n")


def save_prediction_frame_jsonl(predictions: pd.DataFrame, path: Path) -> None:
    ensure_parent_dir(path)
    with path.open("w", encoding="utf-8") as file:
        for row in predictions.itertuples(index=False):
            record = {
                "itemId": int(row.itemId),
                "detectedMcIds": list(row.predDetectedMcIds),
                "shouldSplit": bool(row.predShouldSplit),
                "drafts": list(row.predDrafts),
            }
            file.write(json.dumps(record, ensure_ascii=False) + "\n")


def prediction_results_to_frame(results: list[PredictionResult]) -> pd.DataFrame:
    records: list[dict[str, object]] = []
    for result in results:
        records.append(
            {
                "itemId": result.itemId,
                "predDetectedMcIds": result.detectedMcIds,
                "predShouldSplit": result.shouldSplit,
                "predDrafts": [draft.__dict__ for draft in result.drafts],
                "predCategoryDecisions": [decision.__dict__ for decision in result.categoryDecisions],
            }
        )
    return pd.DataFrame(records)
