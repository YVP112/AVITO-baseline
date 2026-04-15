from __future__ import annotations

from dataclasses import asdict

import pandas as pd

from src.contracts import DraftPrediction, PredictionResult
from src.draft_generator import generate_draft_text
from src.ml_baseline import MLBaselinePredictor
from src.text_preprocessing import normalize_text


FINAL_MODEL_VERSION = "tfidf-logreg-services-splitter-v1"

TURNKEY_SCOPE_MARKERS = (
    "под ключ в том числе",
    "под ключ от",
    "с полным сопровождением объекта",
    "полным сопровождением объекта",
    "полный цикл работ",
    "полного цикла работ",
    "полный спектр работ",
    "весь спектр работ",
    "что входит в наши услуги",
    "входят в наши услуги",
)

STANDALONE_MARKERS = (
    "по отдельности",
    "отдельно",
    "как отдельную",
    "отдельную услугу",
    "отдельные услуги",
    "отдельные виды работ",
    "так же",
    "также",
    "частичный ремонт",
    "частично",
    "мелкий ремонт",
    "муж на час",
    "мастер на час",
    "выполняем работы",
    "оказываем услуги",
)

FORCE_SERVICE_LIST_MARKERS = (
    "мы осуществляем",
    "вот список выполняемых",
    "беремся как за частичный",
    "берем как за частичный",
    "мелкий бытовой ремонт",
    "срочные мелкие ремонтные работы",
)


def blank_case_type(df: pd.DataFrame) -> pd.DataFrame:
    prepared = df.copy()
    prepared["caseType"] = ""
    return prepared


def build_inference_frame(
    item_id: int,
    source_mc_id: int,
    source_mc_title: str,
    description: str,
) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "itemId": int(item_id),
                "sourceMcId": int(source_mc_id),
                "sourceMcTitle": str(source_mc_title),
                "description": str(description),
                "caseType": "",
            }
        ]
    )


def task_response_from_prediction(row: object) -> dict[str, object]:
    return {
        "itemId": int(row.itemId),
        "detectedMcIds": [int(mc_id) for mc_id in row.predDetectedMcIds],
        "shouldSplit": bool(row.predShouldSplit),
        "drafts": list(row.predDrafts),
    }


class FinalSplitterModel:
    def __init__(self, predictor: MLBaselinePredictor) -> None:
        self.predictor = predictor
        self.micro_map = {
            int(row.mcId): str(row.mcTitle)
            for row in predictor.micro_df.itertuples(index=False)
        }

    @classmethod
    def from_training_data(
        cls,
        train_df: pd.DataFrame,
        micro_df: pd.DataFrame,
    ) -> "FinalSplitterModel":
        predictor = MLBaselinePredictor.from_training_data(train_df, micro_df)
        return cls(predictor)

    def predict_dataframe(self, df: pd.DataFrame) -> pd.DataFrame:
        prepared = df.copy()
        if "caseType" not in prepared.columns:
            prepared = blank_case_type(prepared)
        predictions = self.predictor.predict_dataframe(prepared)
        return self.apply_postprocessing(prepared, predictions)

    def predict_item(
        self,
        item_id: int,
        source_mc_id: int,
        description: str,
    ) -> PredictionResult:
        source_mc_title = self.micro_map.get(int(source_mc_id), "")
        inference_df = build_inference_frame(item_id, source_mc_id, source_mc_title, description)
        prediction = self.predict_dataframe(inference_df).iloc[0]
        drafts = [
            DraftPrediction(**draft)
            for draft in prediction.predDrafts
        ]

        return PredictionResult(
            itemId=int(prediction.itemId),
            detectedMcIds=[int(mc_id) for mc_id in prediction.predDetectedMcIds],
            shouldSplit=bool(prediction.predShouldSplit),
            drafts=drafts,
            splitScores={},
            categoryDecisions=[],
        )

    def apply_postprocessing(
        self,
        df: pd.DataFrame,
        predictions: pd.DataFrame,
    ) -> pd.DataFrame:
        processed = predictions.copy(deep=True)

        for index, source_row in df.reset_index(drop=True).iterrows():
            description = str(source_row["description"])
            if self.should_force_service_list(description) and len(processed.at[index, "predDetectedMcIds"]) >= 3:
                split_mc_ids = list(processed.at[index, "predDetectedMcIds"])
                processed.at[index, "predShouldSplit"] = True
                processed.at[index, "predDrafts"] = self.build_drafts_for_mc_ids(split_mc_ids)
                continue

            if not bool(processed.at[index, "predShouldSplit"]):
                continue

            if self.should_suppress_turnkey_scope(description):
                processed.at[index, "predShouldSplit"] = False
                processed.at[index, "predDrafts"] = []

        return processed

    def should_suppress_turnkey_scope(self, description: str) -> bool:
        text = normalize_text(description)
        has_turnkey_scope = any(marker in text for marker in TURNKEY_SCOPE_MARKERS)
        has_standalone_signal = any(marker in text for marker in STANDALONE_MARKERS)
        return has_turnkey_scope and not has_standalone_signal

    def should_force_service_list(self, description: str) -> bool:
        text = normalize_text(description)
        return any(marker in text for marker in FORCE_SERVICE_LIST_MARKERS)

    def build_drafts_for_mc_ids(self, mc_ids: list[int]) -> list[dict[str, object]]:
        drafts: list[dict[str, object]] = []
        for mc_id in mc_ids:
            mc_title = self.micro_map.get(int(mc_id))
            if mc_title is None:
                continue
            draft = DraftPrediction(
                mcId=int(mc_id),
                mcTitle=mc_title,
                text=generate_draft_text(mc_title, []),
            )
            drafts.append(asdict(draft))
        return drafts
