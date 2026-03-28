from __future__ import annotations

import json
from pathlib import Path

from src.config import DATASET_PATH, MICRO_PATH, OUTPUTS_DIR
from src.data_loader import load_dataset
from src.evaluate import evaluate_predictions
from src.predictor import BaselinePredictor


def save_predictions(df, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False, encoding="utf-8-sig")


def save_predictions_jsonl(df, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    records = []
    for row in df.itertuples(index=False):
        records.append(
            {
                "itemId": int(row.itemId),
                "detectedMcIds": row.predDetectedMcIds,
                "shouldSplit": bool(row.predShouldSplit),
                "drafts": row.predDrafts,
            }
        )

    with path.open("w", encoding="utf-8") as file:
        for record in records:
            file.write(json.dumps(record, ensure_ascii=False) + "\n")


def main() -> None:
    dataset_df = load_dataset(DATASET_PATH)
    predictor = BaselinePredictor.from_csv(MICRO_PATH)

    val_df = dataset_df[dataset_df["split"] == "val"].copy().reset_index(drop=True)
    test_df = dataset_df[dataset_df["split"] == "test"].copy().reset_index(drop=True)

    val_predictions = predictor.predict_dataframe(val_df)
    test_predictions = predictor.predict_dataframe(test_df)

    val_result = val_df.merge(val_predictions, on="itemId", how="left")
    test_result = test_df.merge(test_predictions, on="itemId", how="left")

    val_metrics = evaluate_predictions(val_result)

    save_predictions(val_result, OUTPUTS_DIR / "val_predictions.csv")
    save_predictions(test_result, OUTPUTS_DIR / "test_predictions.csv")
    save_predictions_jsonl(val_result, OUTPUTS_DIR / "val_predictions.jsonl")
    save_predictions_jsonl(test_result, OUTPUTS_DIR / "test_predictions.jsonl")

    print("Validation metrics:")
    for metric_name, value in val_metrics.items():
        print(f"  {metric_name}: {value:.4f}")

    print()
    print(f"Saved validation predictions to: {OUTPUTS_DIR / 'val_predictions.csv'}")
    print(f"Saved test predictions to: {OUTPUTS_DIR / 'test_predictions.csv'}")
    print(f"Saved validation predictions to: {OUTPUTS_DIR / 'val_predictions.jsonl'}")
    print(f"Saved test predictions to: {OUTPUTS_DIR / 'test_predictions.jsonl'}")


if __name__ == "__main__":
    main()
