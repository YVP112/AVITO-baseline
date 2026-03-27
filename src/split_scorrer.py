from __future__ import annotations

from src.config import (
    DIRECT_SPLIT_MARKER_BONUS,
    MULTI_PHRASE_BONUS,
    NO_DIRECT_SPLIT_WITH_STRONG_NEGATIVE_PENALTY,
    SPLIT_SCORE_THRESHOLD,
    STRONG_NEGATIVE_SPLIT_MARKERS,
    STRONG_POSITIVE_SPLIT_MARKERS,
    TEXT_WIDE_STRONG_NEGATIVE_PENALTY,
    TEXT_WIDE_STRONG_POSITIVE_BONUS,
    TEXT_WIDE_WEAK_NEGATIVE_PENALTY,
    TEXT_WIDE_WEAK_POSITIVE_BONUS,
    TITLE_PHRASE_BONUS,
    WEAK_NEGATIVE_SPLIT_MARKERS,
    WEAK_POSITIVE_SPLIT_MARKERS,
)
from src.text_preprocessing import normalize_text, split_sentences


def _count_markers(sentence: str, markers: tuple[str, ...]) -> int:
    return sum(1 for marker in markers if marker in sentence)


def _has_direct_split_marker(sentence: str) -> bool:
    return any(marker in sentence for marker in STRONG_POSITIVE_SPLIT_MARKERS)


def _sentence_score(sentence: str, matched_phrases: list[str]) -> tuple[float, bool]:
    score = 0.0
    normalized_sentence = normalize_text(sentence)

    phrase_hits = 0
    for phrase in matched_phrases:
        if phrase and phrase in normalized_sentence:
            phrase_hits += 1

    score += 0.6 * phrase_hits
    if phrase_hits >= 2:
        score += MULTI_PHRASE_BONUS

    strong_positive_hits = _count_markers(normalized_sentence, STRONG_POSITIVE_SPLIT_MARKERS)
    weak_positive_hits = _count_markers(normalized_sentence, WEAK_POSITIVE_SPLIT_MARKERS)
    strong_negative_hits = _count_markers(normalized_sentence, STRONG_NEGATIVE_SPLIT_MARKERS)
    weak_negative_hits = _count_markers(normalized_sentence, WEAK_NEGATIVE_SPLIT_MARKERS)

    if strong_positive_hits:
        score += DIRECT_SPLIT_MARKER_BONUS
        score += 0.5 * (strong_positive_hits - 1)

    score += 0.2 * weak_positive_hits
    score -= 1.0 * strong_negative_hits
    score -= 0.25 * weak_negative_hits

    return score, _has_direct_split_marker(normalized_sentence)


def score_microcategory_split(
    description: str,
    mc_title: str,
    matched_phrases: list[str],
) -> float:
    normalized_text = normalize_text(description)
    normalized_title = normalize_text(mc_title)
    sentences = split_sentences(description)

    sentence_results = [
        _sentence_score(sentence, matched_phrases)
        for sentence in sentences
        if any(phrase in sentence for phrase in matched_phrases)
    ]

    sentence_scores = [score for score, _ in sentence_results]
    has_direct_local_signal = any(has_direct for _, has_direct in sentence_results)
    score = max(sentence_scores) if sentence_scores else 0.0

    if normalized_title and normalized_title in normalized_text:
        score += TITLE_PHRASE_BONUS

    if len(matched_phrases) >= 2:
        score += MULTI_PHRASE_BONUS

    if any(marker in normalized_text for marker in STRONG_POSITIVE_SPLIT_MARKERS):
        score += TEXT_WIDE_STRONG_POSITIVE_BONUS

    if any(marker in normalized_text for marker in WEAK_POSITIVE_SPLIT_MARKERS):
        score += TEXT_WIDE_WEAK_POSITIVE_BONUS

    has_strong_negative = any(marker in normalized_text for marker in STRONG_NEGATIVE_SPLIT_MARKERS)
    if has_strong_negative:
        score -= TEXT_WIDE_STRONG_NEGATIVE_PENALTY

    if any(marker in normalized_text for marker in WEAK_NEGATIVE_SPLIT_MARKERS):
        score -= TEXT_WIDE_WEAK_NEGATIVE_PENALTY

    if has_strong_negative and not has_direct_local_signal:
        score -= NO_DIRECT_SPLIT_WITH_STRONG_NEGATIVE_PENALTY

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
