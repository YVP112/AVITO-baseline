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
    "можно заказать отдельно",
    "как отдельную услугу",
    "как самостоятельную работу",
    "самостоятельную работу",
    "беру как самостоятельную работу",
    "выполняем отдельно",
    "делаем отдельно",
    "также",
    "кроме того",
)

WEAK_POSITIVE_SPLIT_MARKERS = (
    "еще",
    "плюс",
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
    "комплекс",
    "комплексный",
    "комплекс работ",
    "весь цикл",
    "под ключ с",
    "от демонтажа до чистовой",
    "объект целиком",
    "целиком",
    "по отдельным видам работ не выезжаю",
    "без дробления на этапы",
    "именно на комплекс",
)

WEAK_NEGATIVE_SPLIT_MARKERS = (
    "в рамках",
    "часть ремонта",
    "этапы",
    "в комплексе",
)

SPLIT_SCORE_THRESHOLD = 1.3
TEXT_WIDE_STRONG_POSITIVE_BONUS = 0.45
TEXT_WIDE_WEAK_POSITIVE_BONUS = 0.15
TEXT_WIDE_STRONG_NEGATIVE_PENALTY = 0.55
TEXT_WIDE_WEAK_NEGATIVE_PENALTY = 0.2
TITLE_PHRASE_BONUS = 0.25
MULTI_PHRASE_BONUS = 0.3
DIRECT_MARKER_BONUS = 0.9
SEPARATOR_ENUM_BONUS = 0.45
SHORT_TEXT_MULTI_MATCH_BONUS = 0.4
MAX_DRAFT_PHRASES = 3
