from __future__ import annotations

from pathlib import Path

import pandas as pd

from src.config import DATASET_PATH, MICRO_PATH, OUTPUTS_DIR
from src.data_loader import load_dataset, load_microcategories
from src.evaluate import evaluate_predictions
from src.ml_baseline import MLBaselinePredictor
from src.predictor import BaselinePredictor


def save_metrics_table(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False, encoding="utf-8-sig")


def main() -> None:
    dataset_df = load_dataset(DATASET_PATH)
    micro_df = load_microcategories(MICRO_PATH)

    train_df = dataset_df[dataset_df["split"] == "train"].copy().reset_index(drop=True)
    val_df = dataset_df[dataset_df["split"] == "val"].copy().reset_index(drop=True)

    heuristic_predictor = BaselinePredictor(micro_df)
    heuristic_predictions = heuristic_predictor.predict_dataframe(val_df)
    heuristic_result = val_df.merge(heuristic_predictions, on="itemId", how="left")
    heuristic_metrics = evaluate_predictions(heuristic_result)

    ml_predictor = MLBaselinePredictor.from_training_data(train_df, micro_df)
    ml_predictions = ml_predictor.predict_dataframe(val_df)
    ml_result = val_df.merge(ml_predictions, on="itemId", how="left")
    ml_metrics = evaluate_predictions(ml_result)

    comparison_df = pd.DataFrame(
        [
            {"baseline": "heuristic", **heuristic_metrics},
            {"baseline": "ml", **ml_metrics},
        ]
    )

    save_metrics_table(comparison_df, OUTPUTS_DIR / "baseline_comparison.csv")

    print("Validation comparison:")
    for row in comparison_df.itertuples(index=False):
        print(f"  {row.baseline}:")
        print(f"    precision_micro: {row.precision_micro:.4f}")
        print(f"    recall_micro: {row.recall_micro:.4f}")
        print(f"    f1_micro: {row.f1_micro:.4f}")
        print(f"    should_split_accuracy: {row.should_split_accuracy:.4f}")

    print()
    print(f"Saved comparison to: {OUTPUTS_DIR / 'baseline_comparison.csv'}")


if __name__ == "__main__":
    main()
