from __future__ import annotations

from src.config import MAX_DRAFT_PHRASES


def generate_draft_text(mc_title: str, matched_phrases: list[str]) -> str:
    phrases = matched_phrases[:MAX_DRAFT_PHRASES]
    if phrases:
        return f"Выполняем {mc_title.lower()} отдельно: " + ", ".join(phrases) + "."
    return f"Выполняем {mc_title.lower()} отдельно. Уточняйте детали и стоимость по объекту."


def generate_drafts(
    split_mc_ids: list[int],
    phrase_index: dict[int, dict[str, object]],
    matched_phrases_by_mc: dict[int, list[str]],
) -> list[dict[str, object]]:
    drafts: list[dict[str, object]] = []

    for mc_id in split_mc_ids:
        mc_title = str(phrase_index[mc_id]["mcTitle"])
        matched_phrases = matched_phrases_by_mc.get(mc_id, [])
        drafts.append(
            {
                "mcId": mc_id,
                "mcTitle": mc_title,
                "text": generate_draft_text(mc_title, matched_phrases),
            }
        )

    return drafts
