from __future__ import annotations

from src.config import DATASET_PATH, MICRO_PATH, OUTPUTS_DIR
from src.data_loader import load_dataset, load_microcategories
from src.evaluate import build_error_analysis_frame, evaluate_predictions
from src.ml_baseline import MLBaselinePredictor
from src.reporting import save_dataframe_csv, save_prediction_frame_jsonl


def main() -> None:
    dataset_df = load_dataset(DATASET_PATH)
    micro_df = load_microcategories(MICRO_PATH)

    train_df = dataset_df[dataset_df["split"] == "train"].copy().reset_index(drop=True)
    val_df = dataset_df[dataset_df["split"] == "val"].copy().reset_index(drop=True)
    test_df = dataset_df[dataset_df["split"] == "test"].copy().reset_index(drop=True)

    predictor = MLBaselinePredictor.from_training_data(train_df, micro_df)
    predictor.apply_public_validation_threshold_profile()
    val_predictions = predictor.predict_dataframe(val_df)
    test_predictions = predictor.predict_dataframe(test_df)

    val_result = val_df.merge(val_predictions, on="itemId", how="left")
    test_result = test_df.merge(test_predictions, on="itemId", how="left")

    val_metrics = evaluate_predictions(val_result)
    val_error_analysis = build_error_analysis_frame(val_result)

    save_dataframe_csv(val_result, OUTPUTS_DIR / "val_predictions.csv")
    save_dataframe_csv(test_result, OUTPUTS_DIR / "test_predictions.csv")
    save_dataframe_csv(val_error_analysis, OUTPUTS_DIR / "error_analysis.csv")
    save_prediction_frame_jsonl(val_predictions, OUTPUTS_DIR / "val_predictions.jsonl")
    save_prediction_frame_jsonl(test_predictions, OUTPUTS_DIR / "test_predictions.jsonl")

    print("Validation metrics:")
    for metric_name, value in val_metrics.items():
        print(f"  {metric_name}: {value:.4f}")

    print()
    print(f"Saved validation predictions to: {OUTPUTS_DIR / 'val_predictions.csv'}")
    print(f"Saved test predictions to: {OUTPUTS_DIR / 'test_predictions.csv'}")
    print(f"Saved error analysis to: {OUTPUTS_DIR / 'error_analysis.csv'}")
    print(f"Saved validation predictions to: {OUTPUTS_DIR / 'val_predictions.jsonl'}")
    print(f"Saved test predictions to: {OUTPUTS_DIR / 'test_predictions.jsonl'}")


if __name__ == "__main__":
    main()
