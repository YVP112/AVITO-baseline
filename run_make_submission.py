from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import pandas as pd

from src.config import DATASET_PATH, MICRO_PATH, OUTPUTS_DIR
from src.data_loader import load_dataset, load_microcategories
from src.final_model import FinalSplitterModel, task_response_from_prediction


DEFAULT_TEST_PATH = Path("rnc_test.csv")
SUBMISSION_PATH = OUTPUTS_DIR / "rnc_test_responses.csv"
AUDIT_PATH = OUTPUTS_DIR / "rnc_test_audit.csv"


def resolve_test_path(argv: list[str]) -> Path:
    if len(argv) > 1:
        return Path(argv[1])
    return Path(os.getenv("RNC_TEST_PATH", str(DEFAULT_TEST_PATH)))


def load_requests(test_path: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    raw_test_df = pd.read_csv(test_path)
    request_df = pd.DataFrame([json.loads(text) for text in raw_test_df["request"]])
    return raw_test_df, request_df


def request_to_inference_frame(request_df: pd.DataFrame) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "itemId": request_df["itemId"].astype(int),
            "sourceMcId": request_df["mcId"].astype(int),
            "sourceMcTitle": request_df["mcTitle"].astype(str),
            "description": request_df["description"].astype(str),
            "caseType": "",
        }
    )


def save_submission(
    raw_test_df: pd.DataFrame,
    predictions: pd.DataFrame,
    submission_path: Path,
) -> None:
    responses = [
        json.dumps(task_response_from_prediction(row), ensure_ascii=False)
        for row in predictions.itertuples(index=False)
    ]

    submission = raw_test_df.copy()
    submission["response"] = responses
    submission_path.parent.mkdir(parents=True, exist_ok=True)
    submission.to_csv(submission_path, index=False, encoding="utf-8-sig")


def save_audit(
    request_df: pd.DataFrame,
    predictions: pd.DataFrame,
    audit_path: Path,
) -> None:
    audit = pd.DataFrame(
        {
            "itemId": request_df["itemId"],
            "descriptionShort": request_df["description"].str.replace("\n", " ", regex=False).str[:700],
            "shouldSplit": predictions["predShouldSplit"],
            "detectedMcIds": predictions["predDetectedMcIds"],
            "drafts": predictions["predDrafts"],
        }
    )
    audit_path.parent.mkdir(parents=True, exist_ok=True)
    audit.to_csv(audit_path, index=False, encoding="utf-8-sig")


def print_prediction_stats(predictions: pd.DataFrame) -> None:
    split_count = int(predictions["predShouldSplit"].sum())
    draft_count = int(predictions["predDrafts"].apply(len).sum())
    print(f"Submission rows: {len(predictions)}")
    print(f"Predicted split rows: {split_count}")
    print(f"Predicted draft count: {draft_count}")


def main() -> None:
    test_path = resolve_test_path(sys.argv)
    if not test_path.exists():
        raise FileNotFoundError(f"Test file not found: {test_path}")

    raw_test_df, request_df = load_requests(test_path)
    inference_df = request_to_inference_frame(request_df)

    train_df = load_dataset(DATASET_PATH).copy().reset_index(drop=True)
    train_df["caseType"] = ""
    micro_df = load_microcategories(MICRO_PATH)

    model = FinalSplitterModel.from_training_data(train_df, micro_df)
    predictions = model.predict_dataframe(inference_df)

    save_submission(raw_test_df, predictions, SUBMISSION_PATH)
    save_audit(request_df, predictions, AUDIT_PATH)
    print_prediction_stats(predictions)
    print(f"Saved submission to: {SUBMISSION_PATH}")
    print(f"Saved audit to: {AUDIT_PATH}")


if __name__ == "__main__":
    main()
