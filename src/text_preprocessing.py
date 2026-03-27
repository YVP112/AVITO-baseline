from __future__ import annotations

import re


MULTISPACE_RE = re.compile(r"\s+")
NON_WORD_RE = re.compile(r"[^a-zа-я0-9\s]+", flags=re.IGNORECASE)


def normalize_text(text: str) -> str:
    """Lowercase text, replace 'ё', drop punctuation, and normalize spaces."""
    if text is None:
        return ""

    normalized = str(text).lower()
    normalized = normalized.replace("ё", "е")
    normalized = NON_WORD_RE.sub(" ", normalized)
    normalized = MULTISPACE_RE.sub(" ", normalized).strip()
    return normalized


def split_sentences(text: str) -> list[str]:
    """Split raw text into simple sentence-like chunks, including bullets and separators."""
    if text is None:
        return []

    chunks = re.split(r"[.!?;\n\r]|(?:\s[-•]\s)|/|\+", str(text))
    return [normalize_text(chunk) for chunk in chunks if normalize_text(chunk)]


def contains_whole_phrase(text: str, phrase: str) -> bool:
    """Check phrase presence with rough word boundaries."""
    normalized_text = f" {normalize_text(text)} "
    normalized_phrase = normalize_text(phrase)
    if not normalized_phrase:
        return False
    return f" {normalized_phrase} " in normalized_text
