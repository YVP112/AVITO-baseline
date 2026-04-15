from __future__ import annotations

import re


MULTISPACE_RE = re.compile(r"\s+")
NON_WORD_RE = re.compile(r"[^a-zа-я0-9\s]+", flags=re.IGNORECASE)
TOKEN_ENDINGS = (
    "иями",
    "ями",
    "ами",
    "его",
    "ого",
    "ему",
    "ому",
    "ыми",
    "ими",
    "ий",
    "ый",
    "ой",
    "ая",
    "яя",
    "ое",
    "ее",
    "ые",
    "ие",
    "ов",
    "ев",
    "ей",
    "ых",
    "их",
    "ам",
    "ям",
    "ах",
    "ях",
    "ом",
    "ем",
    "у",
    "ю",
    "а",
    "я",
    "ы",
    "и",
    "е",
)
SEO_MARKERS = (
    "меня можно найти по запросам",
    "возможно вы искали нас по запросам",
    "возможно вы искали",
    "нас ищут по таким запросам",
    "нас ищут по запросам",
    "вы искали нас по запросам",
    "ключевые слова",
)


def normalize_text(text: str) -> str:
    """Lowercase text, replace 'ё', drop punctuation, and normalize spaces."""
    if text is None:
        return ""

    normalized = str(text).lower()
    normalized = normalized.replace("ё", "е")
    normalized = NON_WORD_RE.sub(" ", normalized)
    normalized = MULTISPACE_RE.sub(" ", normalized).strip()
    return normalized


def strip_seo_noise(text: str) -> str:
    if text is None:
        return ""

    raw_text = str(text)
    lowered = raw_text.lower().replace("ё", "е")
    cut_position = len(raw_text)

    for marker in SEO_MARKERS:
        marker_position = lowered.find(marker)
        if marker_position != -1:
            cut_position = min(cut_position, marker_position)

    cleaned = raw_text[:cut_position]
    return cleaned.strip()


def split_sentences(text: str) -> list[str]:
    """Split raw text into simple sentence-like chunks, including bullets and separators."""
    if text is None:
        return []

    chunks = re.split(r"[.!?;\n\r]|(?:\s[-•]\s)|/|\+", str(text))
    return [normalize_text(chunk) for chunk in chunks if normalize_text(chunk)]


def tokenize_text(text: str) -> list[str]:
    normalized = normalize_text(text)
    return normalized.split() if normalized else []


def stem_token(token: str) -> str:
    normalized = normalize_text(token)
    if len(normalized) <= 4:
        return normalized

    for ending in TOKEN_ENDINGS:
        if normalized.endswith(ending) and len(normalized) - len(ending) >= 4:
            return normalized[: -len(ending)]

    return normalized


def stem_tokens(tokens: list[str]) -> list[str]:
    return [stem_token(token) for token in tokens if token]


def contains_stemmed_phrase(text: str, phrase: str) -> bool:
    text_tokens = tokenize_text(text)
    phrase_tokens = tokenize_text(phrase)
    if not text_tokens or not phrase_tokens or len(phrase_tokens) > len(text_tokens):
        return False

    text_roots = stem_tokens(text_tokens)
    phrase_roots = stem_tokens(phrase_tokens)
    window_size = len(phrase_roots)

    for start in range(len(text_roots) - window_size + 1):
        if text_roots[start : start + window_size] == phrase_roots:
            return True
    return False


def contains_whole_phrase(text: str, phrase: str) -> bool:
    """Check phrase presence with rough word boundaries."""
    normalized_text = f" {normalize_text(text)} "
    normalized_phrase = normalize_text(phrase)
    if not normalized_phrase:
        return False
    if f" {normalized_phrase} " in normalized_text:
        return True
    return contains_stemmed_phrase(text, phrase)
