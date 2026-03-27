from __future__ import annotations

from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data" / "raw"
OUTPUTS_DIR = PROJECT_ROOT / "outputs"

DATASET_PATH = DATA_DIR / "rnc_dataset.csv"
MICRO_PATH = DATA_DIR / "rnc_mic_key_phrases.csv"

STRONG_POSITIVE_SPLIT_MARKERS = (
    "отдельно",
    "по отдельности",
    "отдельная услуга",
    "отдельные услуги",
    "отдельно выполняем",
    "отдельно делаем",
    "также",
    "кроме того",
    "плюс",
)

WEAK_POSITIVE_SPLIT_MARKERS = (
    "выполняем",
    "делаем",
    "предлагаем",
    "оказываем",
    "услуги",
    "выезд",
    "вызов",
)

STRONG_NEGATIVE_SPLIT_MARKERS = (
    "включая",
    "в составе",
    "под ключ",
    "под ключ с",
    "комплекс",
    "комплексный",
    "комплекс работ",
    "весь цикл",
    "от демонтажа до чистовой",
    "объект целиком",
    "целиком",
    "в рамках",
    "входит в",
    "включено в",
    "с инженерией и отделкой",
)

WEAK_NEGATIVE_SPLIT_MARKERS = (
    "подготовка",
    "этап работ",
    "черновой этап",
    "часть ремонта",
)

SPLIT_SCORE_THRESHOLD = 1.6
TEXT_WIDE_STRONG_POSITIVE_BONUS = 0.4
TEXT_WIDE_WEAK_POSITIVE_BONUS = 0.15
TEXT_WIDE_STRONG_NEGATIVE_PENALTY = 0.9
TEXT_WIDE_WEAK_NEGATIVE_PENALTY = 0.3
TITLE_PHRASE_BONUS = 0.25
MULTI_PHRASE_BONUS = 0.35
DIRECT_SPLIT_MARKER_BONUS = 1.25
NO_DIRECT_SPLIT_WITH_STRONG_NEGATIVE_PENALTY = 0.9
MAX_DRAFT_PHRASES = 3
