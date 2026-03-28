from __future__ import annotations

import ast
from pathlib import Path
from typing import Iterable

import pandas as pd


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


def load_dataset(path: str | Path) -> pd.DataFrame:
    path = Path(path)
    df = pd.read_csv(path)

    _validate_columns(df, DATASET_REQUIRED_COLUMNS, path)

    df["targetDetectedMcIds"] = df["targetDetectedMcIds"].apply(parse_id_list)
    df["targetSplitMcIds"] = df["targetSplitMcIds"].apply(parse_id_list)
    df["shouldSplit"] = df["shouldSplit"].apply(parse_bool)

    df["itemId"] = df["itemId"].astype(int)
    df["sourceMcId"] = df["sourceMcId"].astype(int)
    df["sourceMcTitle"] = df["sourceMcTitle"].fillna("").astype(str)
    df["description"] = df["description"].fillna("").astype(str)
    df["caseType"] = df["caseType"].fillna("").astype(str)
    df["split"] = df["split"].fillna("").astype(str)

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
