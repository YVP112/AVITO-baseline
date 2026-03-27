from __future__ import annotations

import pandas as pd

from src.data_loader import load_dataset, load_microcategories
from src.dictionary_matcher import build_phrase_index, detect_microcategories, group_matches_by_mc
from src.draft_generator import generate_drafts
from src.split_scorrer import resolve_split_mc_ids


class BaselinePredictor:
    def __init__(self, micro_df: pd.DataFrame) -> None:
        self.micro_df = micro_df
        self.phrase_index = build_phrase_index(micro_df)

    @classmethod
    def from_csv(cls, micro_path: str) -> "BaselinePredictor":
        micro_df = load_microcategories(micro_path)
        return cls(micro_df)

    def predict_item(
        self,
        item_id: int,
        source_mc_id: int,
        description: str,
    ) -> dict[str, object]:
        detected_mc_ids = detect_microcategories(description, self.phrase_index)
        matched_phrases_by_mc = group_matches_by_mc(description, self.phrase_index)
        split_mc_ids, split_scores = resolve_split_mc_ids(
            description=description,
            source_mc_id=source_mc_id,
            phrase_index=self.phrase_index,
            matched_phrases_by_mc=matched_phrases_by_mc,
        )
        drafts = generate_drafts(split_mc_ids, self.phrase_index, matched_phrases_by_mc)

        return {
            "itemId": item_id,
            "detectedMcIds": detected_mc_ids,
            "shouldSplit": bool(split_mc_ids),
            "splitMcIds": split_mc_ids,
            "drafts": drafts,
            "splitScores": split_scores,
        }

    def predict_dataframe(self, df: pd.DataFrame) -> pd.DataFrame:
        records: list[dict[str, object]] = []
        for row in df.itertuples(index=False):
            prediction = self.predict_item(
                item_id=int(row.itemId),
                source_mc_id=int(row.sourceMcId),
                description=str(row.description),
            )
            records.append(
                {
                    "itemId": prediction["itemId"],
                    "predDetectedMcIds": prediction["detectedMcIds"],
                    "predShouldSplit": prediction["shouldSplit"],
                    "predSplitMcIds": prediction["splitMcIds"],
                    "predDrafts": prediction["drafts"],
                }
            )
        return pd.DataFrame(records)


def build_predictor(micro_path: str) -> BaselinePredictor:
    return BaselinePredictor.from_csv(micro_path)


def predict_from_paths(dataset_path: str, micro_path: str) -> pd.DataFrame:
    dataset_df = load_dataset(dataset_path)
    predictor = build_predictor(micro_path)
    predictions_df = predictor.predict_dataframe(dataset_df)
    return dataset_df.merge(predictions_df, on="itemId", how="left")
