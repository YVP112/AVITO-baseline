from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from src.text_preprocessing import contains_whole_phrase, normalize_text


@dataclass(frozen=True)
class PhraseMatch:
    mc_id: int
    mc_title: str
    phrase: str


def split_key_phrases(key_phrases: str) -> list[str]:
    phrases = [normalize_text(part) for part in str(key_phrases).split(";")]
    return [phrase for phrase in phrases if phrase]


def build_phrase_index(micro_df: pd.DataFrame) -> dict[int, dict[str, object]]:
    phrase_index: dict[int, dict[str, object]] = {}
    for row in micro_df.itertuples(index=False):
        phrases = split_key_phrases(row.keyPhrases)
        phrase_index[int(row.mcId)] = {
            "mcId": int(row.mcId),
            "mcTitle": str(row.mcTitle),
            "description": str(row.description),
            "phrases": phrases,
        }
    return phrase_index


def find_phrase_matches(text: str, phrase_index: dict[int, dict[str, object]]) -> list[PhraseMatch]:
    matches: list[PhraseMatch] = []
    normalized_text = normalize_text(text)

    for mc_id, payload in phrase_index.items():
        mc_title = str(payload["mcTitle"])
        for phrase in payload["phrases"]:
            if contains_whole_phrase(normalized_text, phrase):
                matches.append(PhraseMatch(mc_id=mc_id, mc_title=mc_title, phrase=phrase))

    return matches


def detect_microcategories(text: str, phrase_index: dict[int, dict[str, object]]) -> list[int]:
    matches = find_phrase_matches(text, phrase_index)
    return sorted({match.mc_id for match in matches})


def group_matches_by_mc(text: str, phrase_index: dict[int, dict[str, object]]) -> dict[int, list[str]]:
    grouped: dict[int, list[str]] = {}
    for match in find_phrase_matches(text, phrase_index):
        grouped.setdefault(match.mc_id, [])
        if match.phrase not in grouped[match.mc_id]:
            grouped[match.mc_id].append(match.phrase)
    return grouped
