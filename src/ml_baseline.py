from __future__ import annotations

from dataclasses import dataclass

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.multiclass import OneVsRestClassifier
from sklearn.preprocessing import MultiLabelBinarizer

from src.data_loader import load_microcategories
from src.draft_generator import generate_draft_text


@dataclass
class MLArtifacts:
    vectorizer: TfidfVectorizer
    detected_mlb: MultiLabelBinarizer
    split_mlb: MultiLabelBinarizer
    detected_model: OneVsRestClassifier
    split_model: OneVsRestClassifier
    should_split_model: LogisticRegression


class MLBaselinePredictor:
    def __init__(self, micro_df: pd.DataFrame, artifacts: MLArtifacts) -> None:
        self.micro_df = micro_df
        self.artifacts = artifacts
        self.micro_map = {
            int(row.mcId): {
                "mcTitle": str(row.mcTitle),
                "description": str(row.description),
            }
            for row in micro_df.itertuples(index=False)
        }

    @classmethod
    def from_training_data(
        cls,
        train_df: pd.DataFrame,
        micro_df: pd.DataFrame,
    ) -> "MLBaselinePredictor":
        train_texts = cls._build_texts(train_df)

        vectorizer = TfidfVectorizer(
            lowercase=True,
            ngram_range=(1, 2),
            min_df=2,
            max_features=30000,
            sublinear_tf=True,
        )
        x_train = vectorizer.fit_transform(train_texts)

        all_mc_ids = sorted(int(mc_id) for mc_id in micro_df["mcId"].tolist())

        detected_mlb = MultiLabelBinarizer(classes=all_mc_ids)
        y_detected = detected_mlb.fit_transform(train_df["targetDetectedMcIds"])
        detected_model = OneVsRestClassifier(
            LogisticRegression(max_iter=1200, class_weight="balanced")
        )
        detected_model.fit(x_train, y_detected)

        split_mlb = MultiLabelBinarizer(classes=all_mc_ids)
        y_split = split_mlb.fit_transform(train_df["targetSplitMcIds"])
        split_model = OneVsRestClassifier(
            LogisticRegression(max_iter=1200, class_weight="balanced")
        )
        split_model.fit(x_train, y_split)

        should_split_model = LogisticRegression(max_iter=1200, class_weight="balanced")
        should_split_model.fit(x_train, train_df["shouldSplit"].astype(int))

        artifacts = MLArtifacts(
            vectorizer=vectorizer,
            detected_mlb=detected_mlb,
            split_mlb=split_mlb,
            detected_model=detected_model,
            split_model=split_model,
            should_split_model=should_split_model,
        )
        return cls(micro_df, artifacts)

    @classmethod
    def from_csv(cls, train_df: pd.DataFrame, micro_path: str) -> "MLBaselinePredictor":
        micro_df = load_microcategories(micro_path)
        return cls.from_training_data(train_df, micro_df)

    @staticmethod
    def _build_texts(df: pd.DataFrame) -> list[str]:
        return [
            f"{row.sourceMcTitle} [SEP] {row.description}"
            for row in df.itertuples(index=False)
        ]

    def _predict_label_lists(
        self,
        x_matrix,
        mlb: MultiLabelBinarizer,
        model: OneVsRestClassifier,
    ) -> list[list[int]]:
        y_pred = model.predict(x_matrix)
        return [sorted(int(label) for label in labels) for labels in mlb.inverse_transform(y_pred)]

    def _build_drafts(self, split_mc_ids: list[int]) -> list[dict[str, object]]:
        drafts: list[dict[str, object]] = []
        for mc_id in split_mc_ids:
            mc_payload = self.micro_map.get(mc_id)
            if mc_payload is None:
                continue
            mc_title = str(mc_payload["mcTitle"])
            drafts.append(
                {
                    "mcId": mc_id,
                    "mcTitle": mc_title,
                    "text": generate_draft_text(mc_title, []),
                }
            )
        return drafts

    def predict_dataframe(self, df: pd.DataFrame) -> pd.DataFrame:
        texts = self._build_texts(df)
        x_matrix = self.artifacts.vectorizer.transform(texts)

        pred_detected_mc_ids = self._predict_label_lists(
            x_matrix,
            self.artifacts.detected_mlb,
            self.artifacts.detected_model,
        )
        raw_pred_split_mc_ids = self._predict_label_lists(
            x_matrix,
            self.artifacts.split_mlb,
            self.artifacts.split_model,
        )
        pred_should_split = self.artifacts.should_split_model.predict(x_matrix).astype(bool)

        records: list[dict[str, object]] = []
        for row, detected_mc_ids, split_mc_ids, should_split in zip(
            df.itertuples(index=False),
            pred_detected_mc_ids,
            raw_pred_split_mc_ids,
            pred_should_split,
        ):
            filtered_split_mc_ids = sorted(
                mc_id
                for mc_id in split_mc_ids
                if mc_id != int(row.sourceMcId)
            )
            if not should_split:
                filtered_split_mc_ids = []

            final_detected_mc_ids = sorted(set(detected_mc_ids) | set(filtered_split_mc_ids))
            drafts = self._build_drafts(filtered_split_mc_ids)

            records.append(
                {
                    "itemId": int(row.itemId),
                    "predDetectedMcIds": final_detected_mc_ids,
                    "predShouldSplit": bool(filtered_split_mc_ids) if should_split else False,
                    "predDrafts": drafts,
                }
            )

        return pd.DataFrame(records)
