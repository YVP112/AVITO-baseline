from __future__ import annotations

import ast
import json
import warnings
from pathlib import Path
from typing import Iterable

import pandas as pd
from sklearn.model_selection import train_test_split

from src.config import DEFAULT_RANDOM_STATE, TEST_SIZE, TRAIN_SIZE, VAL_SIZE


DATASET_REQUIRED_COLUMNS = (
    "itemId",
    "sourceMcId",
    "sourceMcTitle",
    "description",
    "targetDetectedMcIds",
    "targetSplitMcIds",
    "shouldSplit",
    "caseType",
    "split",
)

MICRO_REQUIRED_COLUMNS = (
    "mcId",
    "mcTitle",
    "keyPhrases",
    "description",
)


def parse_id_list(value: object) -> list[int]:
    """Convert values like '[101, 108]' to a list of integers."""
    if value is None or pd.isna(value):
        return []

    if isinstance(value, list):
        return [int(item) for item in value]

    if isinstance(value, tuple):
        return [int(item) for item in value]

    if isinstance(value, int):
        return [int(value)]

    text = str(value).strip()
    if not text:
        return []

    try:
        parsed = ast.literal_eval(text)
    except (SyntaxError, ValueError):
        parsed = None

    if isinstance(parsed, list):
        return [int(item) for item in parsed]
    if isinstance(parsed, tuple):
        return [int(item) for item in parsed]
    if isinstance(parsed, int):
        return [int(parsed)]

    cleaned = text.strip("[]")
    if not cleaned:
        return []

    return [int(part.strip()) for part in cleaned.split(",") if part.strip()]


def parse_bool(value: object) -> bool:
    if isinstance(value, bool):
        return value

    if value is None or pd.isna(value):
        return False

    text = str(value).strip().lower()
    return text in {"true", "1", "yes"}


def _validate_columns(df: pd.DataFrame, required_columns: Iterable[str], path: Path) -> None:
    missing = [column for column in required_columns if column not in df.columns]
    if missing:
        missing_as_text = ", ".join(missing)
        raise ValueError(f"Missing required columns in {path}: {missing_as_text}")


def _read_dataset_file(path: Path) -> pd.DataFrame:
    if path.suffix.lower() == ".json":
        with path.open("r", encoding="utf-8") as file:
            payload = json.load(file)
        return pd.DataFrame(payload)

    with path.open("r", encoding="utf-8") as file:
        header = file.readline()

    delimiter = ";" if header.count(";") > header.count(",") else ","
    return pd.read_csv(path, sep=delimiter, engine="python")


def _assign_default_splits(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        df = df.copy()
        df["split"] = ""
        return df

    records = df.copy().reset_index(drop=True)
    stratify_target = records["shouldSplit"] if records["shouldSplit"].nunique() > 1 else None

    train_df, holdout_df = train_test_split(
        records,
        train_size=TRAIN_SIZE,
        random_state=DEFAULT_RANDOM_STATE,
        stratify=stratify_target,
    )

    holdout_fraction = VAL_SIZE + TEST_SIZE
    val_share_in_holdout = VAL_SIZE / holdout_fraction
    holdout_stratify = holdout_df["shouldSplit"] if holdout_df["shouldSplit"].nunique() > 1 else None
    val_df, test_df = train_test_split(
        holdout_df,
        train_size=val_share_in_holdout,
        random_state=DEFAULT_RANDOM_STATE,
        stratify=holdout_stratify,
    )

    train_df = train_df.copy()
    val_df = val_df.copy()
    test_df = test_df.copy()

    train_df["split"] = "train"
    val_df["split"] = "val"
    test_df["split"] = "test"

    return (
        pd.concat([train_df, val_df, test_df], ignore_index=True)
        .sort_values("itemId")
        .reset_index(drop=True)
    )


def load_dataset(path: str | Path) -> pd.DataFrame:
    path = Path(path)
    df = _read_dataset_file(path)

    _validate_columns(df, DATASET_REQUIRED_COLUMNS, path)

    df["targetDetectedMcIds"] = df["targetDetectedMcIds"].apply(parse_id_list)
    df["targetSplitMcIds"] = df["targetSplitMcIds"].apply(parse_id_list)
    df["shouldSplit"] = df["shouldSplit"].apply(parse_bool)

    df["itemId"] = pd.to_numeric(df["itemId"], errors="coerce")
    df["sourceMcId"] = pd.to_numeric(df["sourceMcId"], errors="coerce")

    invalid_rows = df["itemId"].isna() | df["sourceMcId"].isna()
    if invalid_rows.any():
        dropped_count = int(invalid_rows.sum())
        warnings.warn(
            f"Dropping {dropped_count} malformed rows from dataset {path} due to invalid itemId/sourceMcId",
            stacklevel=2,
        )
        df = df.loc[~invalid_rows].copy()

    df["itemId"] = df["itemId"].astype(int)
    df["sourceMcId"] = df["sourceMcId"].astype(int)
    df["sourceMcTitle"] = df["sourceMcTitle"].fillna("").astype(str)
    df["description"] = df["description"].fillna("").astype(str)
    df["caseType"] = df["caseType"].fillna("").astype(str)
    df["split"] = df["split"].fillna("").astype(str)

    if not df["split"].str.strip().any():
        df = _assign_default_splits(df)

    return df


def load_dataset_split(path: str | Path, split_name: str) -> pd.DataFrame:
    df = load_dataset(path)
    return df[df["split"] == split_name].reset_index(drop=True)


def load_microcategories(path: str | Path) -> pd.DataFrame:
    path = Path(path)
    df = pd.read_csv(path)

    _validate_columns(df, MICRO_REQUIRED_COLUMNS, path)

    df["mcId"] = df["mcId"].astype(int)
    df["mcTitle"] = df["mcTitle"].fillna("").astype(str)
    df["keyPhrases"] = df["keyPhrases"].fillna("").astype(str)
    df["description"] = df["description"].fillna("").astype(str)

    return df
