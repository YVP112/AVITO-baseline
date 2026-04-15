from __future__ import annotations

import os
from functools import lru_cache

from src.config import DATASET_PATH, MICRO_PATH
from src.data_loader import load_dataset, load_microcategories
from src.final_model import FINAL_MODEL_VERSION, FinalSplitterModel


MODEL_VERSION = os.getenv("MODEL_VERSION", FINAL_MODEL_VERSION)
RULES_VERSION = os.getenv("RULES_VERSION", "submission-postprocess-v1")


@lru_cache(maxsize=1)
def get_predictor() -> FinalSplitterModel:
    dataset_df = load_dataset(DATASET_PATH)
    micro_df = load_microcategories(MICRO_PATH)
    return FinalSplitterModel.from_training_data(dataset_df, micro_df)


def get_model_version() -> str:
    return MODEL_VERSION


def get_rules_version() -> str:
    return RULES_VERSION
