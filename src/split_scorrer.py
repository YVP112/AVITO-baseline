from __future__ import annotations

from src.config import (
    NEGATIVE_SPLIT_MARKERS,
    POSITIVE_SPLIT_MARKERS,
    SPLIT_SCORE_THRESHOLD,
    TEXT_WIDE_NEGATIVE_PENALTY,
    TEXT_WIDE_POSITIVE_BONUS,
    TITLE_PHRASE_BONUS,
)
from src.text_preprocessing import normalize_text, split_sentences


def _sentence_score(sentence: str, matched_phrases: list[str]) -> float:
    score = 0.0
    normalized_sentence = normalize_text(sentence)

    for phrase in matched_phrases:
        if phrase and phrase in normalized_sentence:
            score += 1.0

    for marker in POSITIVE_SPLIT_MARKERS:
        if marker in normalized_sentence:
            score += 1.0

    for marker in NEGATIVE_SPLIT_MARKERS:
        if marker in normalized_sentence:
            score -= 1.0

    return score


def score_microcategory_split(
    description: str,
    mc_title: str,
    matched_phrases: list[str],
) -> float:
    normalized_text = normalize_text(description)
    normalized_title = normalize_text(mc_title)
    sentences = split_sentences(description)

    sentence_scores = [
        _sentence_score(sentence, matched_phrases)
        for sentence in sentences
        if any(phrase in sentence for phrase in matched_phrases)
    ]

    score = max(sentence_scores) if sentence_scores else 0.0

    if normalized_title and normalized_title in normalized_text:
        score += TITLE_PHRASE_BONUS

    if any(marker in normalized_text for marker in POSITIVE_SPLIT_MARKERS):
        score += TEXT_WIDE_POSITIVE_BONUS

    if any(marker in normalized_text for marker in NEGATIVE_SPLIT_MARKERS):
        score -= TEXT_WIDE_NEGATIVE_PENALTY

    return score


def score_split_candidates(
    description: str,
    source_mc_id: int,
    phrase_index: dict[int, dict[str, object]],
    matched_phrases_by_mc: dict[int, list[str]],
) -> dict[int, float]:
    scores: dict[int, float] = {}

    for mc_id, matched_phrases in matched_phrases_by_mc.items():
        if mc_id == source_mc_id:
            continue

        mc_title = str(phrase_index[mc_id]["mcTitle"])
        scores[mc_id] = score_microcategory_split(description, mc_title, matched_phrases)

    return scores


def resolve_split_mc_ids(
    description: str,
    source_mc_id: int,
    phrase_index: dict[int, dict[str, object]],
    matched_phrases_by_mc: dict[int, list[str]],
    threshold: float = SPLIT_SCORE_THRESHOLD,
) -> tuple[list[int], dict[int, float]]:
    scores = score_split_candidates(
        description=description,
        source_mc_id=source_mc_id,
        phrase_index=phrase_index,
        matched_phrases_by_mc=matched_phrases_by_mc,
    )
    split_mc_ids = sorted([mc_id for mc_id, score in scores.items() if score >= threshold])
    return split_mc_ids, scores
