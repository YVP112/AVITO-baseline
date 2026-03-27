from __future__ import annotations

from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data" / "raw"
OUTPUTS_DIR = PROJECT_ROOT / "outputs"

DATASET_PATH = DATA_DIR / "rnc_dataset.csv"
MICRO_PATH = DATA_DIR / "rnc_mic_key_phrases.csv"

POSITIVE_SPLIT_MARKERS = (
    "отдельно",
    "также",
    "еще",
    "кроме того",
    "плюс",
    "выполняем",
    "делаем",
    "предлагаем",
    "оказываем",
    "услуги",
    "выезд",
    "вызов",
)

NEGATIVE_SPLIT_MARKERS = (
    "включая",
    "в составе",
    "под ключ",
    "комплекс",
    "комплексный",
    "весь цикл",
    "под ключ с",
    "от демонтажа до чистовой",
    "объект целиком",
    "целиком",
)

SPLIT_SCORE_THRESHOLD = 1.0
TEXT_WIDE_POSITIVE_BONUS = 0.75
TEXT_WIDE_NEGATIVE_PENALTY = 0.5
TITLE_PHRASE_BONUS = 0.25
MAX_DRAFT_PHRASES = 3
