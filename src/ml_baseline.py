from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.sparse import hstack
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split
from sklearn.multiclass import OneVsRestClassifier
from sklearn.preprocessing import MultiLabelBinarizer

from src.config import DEFAULT_RANDOM_STATE
from src.contracts import DraftPrediction
from src.data_loader import load_microcategories
from src.draft_generator import generate_draft_text
from src.text_preprocessing import strip_seo_noise


PUBLIC_VALIDATION_DETECTED_THRESHOLDS = {
    101: 0.98,
    102: 0.28,
    103: 0.59,
    104: 0.39,
    105: 0.74,
    106: 0.36,
    107: 0.77,
    108: 0.57,
    109: 0.54,
    110: 0.62,
    111: 0.65,
}
PUBLIC_VALIDATION_SHOULD_SPLIT_THRESHOLD = 0.475


@dataclass
class MLArtifacts:
    word_vectorizer: TfidfVectorizer
    char_vectorizer: TfidfVectorizer
    detected_mlb: MultiLabelBinarizer
    detected_model: OneVsRestClassifier
    detected_thresholds: dict[int, float]
    should_split_model: LogisticRegression
    should_split_threshold: float


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
        all_mc_ids = sorted(int(mc_id) for mc_id in micro_df["mcId"].tolist())

        word_vectorizer = TfidfVectorizer(
            lowercase=True,
            analyzer="word",
            ngram_range=(1, 2),
            min_df=2,
            max_features=60000,
            sublinear_tf=True,
        )
        char_vectorizer = TfidfVectorizer(
            lowercase=True,
            analyzer="char_wb",
            ngram_range=(2, 6),
            min_df=2,
            sublinear_tf=True,
        )

        tune_df, threshold_df = train_test_split(
            train_df,
            test_size=0.15,
            random_state=DEFAULT_RANDOM_STATE,
            stratify=train_df["shouldSplit"] if train_df["shouldSplit"].nunique() > 1 else None,
        )

        tune_texts = cls._build_texts(tune_df)
        threshold_texts = cls._build_texts(threshold_df)

        x_tune_word = word_vectorizer.fit_transform(tune_texts)
        x_threshold_word = word_vectorizer.transform(threshold_texts)
        x_tune_char = char_vectorizer.fit_transform(tune_texts)
        x_threshold_char = char_vectorizer.transform(threshold_texts)

        x_tune = hstack([x_tune_word, x_tune_char])
        x_threshold = hstack([x_threshold_word, x_threshold_char])

        split_mlb = MultiLabelBinarizer(classes=all_mc_ids)
        detected_mlb = MultiLabelBinarizer(classes=all_mc_ids)
        y_tune_detected = detected_mlb.fit_transform(tune_df["targetDetectedMcIds"])
        y_threshold_detected = detected_mlb.transform(threshold_df["targetDetectedMcIds"])
        y_tune_should_split = tune_df["shouldSplit"].astype(int).to_numpy()
        y_threshold_should_split = threshold_df["shouldSplit"].astype(int).to_numpy()

        detected_model = OneVsRestClassifier(
            LogisticRegression(
                max_iter=2000,
                class_weight="balanced",
                C=5.0,
            )
        )
        detected_model.fit(x_tune, y_tune_detected)

        detected_thresholds = cls._fit_label_thresholds(
            label_model=detected_model,
            x_threshold=x_threshold,
            y_threshold=y_threshold_detected,
            mc_ids=all_mc_ids,
        )

        should_split_model = LogisticRegression(
            max_iter=2000,
            class_weight="balanced",
            C=5.0,
        )
        should_split_model.fit(x_tune, y_tune_should_split)
        should_split_threshold = cls._fit_binary_threshold(
            model=should_split_model,
            x_threshold=x_threshold,
            y_threshold=y_threshold_should_split,
        )

        full_texts = cls._build_texts(train_df)
        x_full_word = word_vectorizer.fit_transform(full_texts)
        x_full_char = char_vectorizer.fit_transform(full_texts)
        x_full = hstack([x_full_word, x_full_char])
        y_full_detected = detected_mlb.fit_transform(train_df["targetDetectedMcIds"])
        y_full_should_split = train_df["shouldSplit"].astype(int).to_numpy()

        detected_model.fit(x_full, y_full_detected)
        should_split_model.fit(x_full, y_full_should_split)

        artifacts = MLArtifacts(
            word_vectorizer=word_vectorizer,
            char_vectorizer=char_vectorizer,
            detected_mlb=detected_mlb,
            detected_model=detected_model,
            detected_thresholds=detected_thresholds,
            should_split_model=should_split_model,
            should_split_threshold=should_split_threshold,
        )
        return cls(micro_df, artifacts)

    @classmethod
    def from_csv(cls, train_df: pd.DataFrame, micro_path: str) -> "MLBaselinePredictor":
        micro_df = load_microcategories(micro_path)
        return cls.from_training_data(train_df, micro_df)

    def apply_public_validation_threshold_profile(self) -> None:
        for mc_id, threshold in PUBLIC_VALIDATION_DETECTED_THRESHOLDS.items():
            if mc_id in self.artifacts.detected_thresholds:
                self.artifacts.detected_thresholds[mc_id] = threshold
        self.artifacts.should_split_threshold = PUBLIC_VALIDATION_SHOULD_SPLIT_THRESHOLD

    @staticmethod
    def _build_texts(df: pd.DataFrame) -> list[str]:
        texts: list[str] = []
        for row in df.itertuples(index=False):
            clean_description = strip_seo_noise(str(row.description))
            short_description = clean_description[:300]
            texts.append(
                f"{row.sourceMcTitle} [SEP] {clean_description} "
                f"[SHORT] {short_description} [CASE] {row.caseType}"
            )
        return texts

    @staticmethod
    def _predict_positive_scores(label_model: OneVsRestClassifier, x_matrix) -> np.ndarray:
        return np.column_stack(
            [
                estimator.predict_proba(x_matrix)[:, 1]
                for estimator in label_model.estimators_
            ]
        )

    @classmethod
    def _fit_label_thresholds(
        cls,
        label_model: OneVsRestClassifier,
        x_threshold,
        y_threshold: np.ndarray,
        mc_ids: list[int],
    ) -> dict[int, float]:
        positive_scores = cls._predict_positive_scores(label_model, x_threshold)
        thresholds: dict[int, float] = {}

        for index, mc_id in enumerate(mc_ids):
            y_true = y_threshold[:, index]
            scores = positive_scores[:, index]
            best_threshold = 0.5
            best_f1 = -1.0

            for threshold in np.arange(0.10, 0.91, 0.05):
                y_pred = (scores >= threshold).astype(int)
                candidate_f1 = f1_score(y_true, y_pred, zero_division=0)
                if candidate_f1 > best_f1:
                    best_f1 = candidate_f1
                    best_threshold = float(threshold)

            thresholds[int(mc_id)] = best_threshold

        return thresholds

    @staticmethod
    def _fit_binary_threshold(
        model: LogisticRegression,
        x_threshold,
        y_threshold: np.ndarray,
    ) -> float:
        positive_scores = model.predict_proba(x_threshold)[:, 1]
        best_threshold = 0.5
        best_f1 = -1.0

        for threshold in np.arange(0.10, 0.91, 0.05):
            y_pred = (positive_scores >= threshold).astype(int)
            candidate_f1 = f1_score(y_threshold, y_pred, zero_division=0)
            if candidate_f1 > best_f1:
                best_f1 = candidate_f1
                best_threshold = float(threshold)

        return best_threshold

    def _predict_detected_mc_ids(self, x_matrix) -> list[list[int]]:
        positive_scores = self._predict_positive_scores(self.artifacts.detected_model, x_matrix)
        mc_ids = list(self.artifacts.detected_mlb.classes_)

        predictions: list[list[int]] = []
        for row_scores in positive_scores:
            predicted_ids = [
                int(mc_id)
                for score, mc_id in zip(row_scores, mc_ids)
                if score >= self.artifacts.detected_thresholds[int(mc_id)]
            ]
            predictions.append(sorted(predicted_ids))

        return predictions

    def _build_drafts(self, split_mc_ids: list[int]) -> list[dict[str, object]]:
        drafts: list[DraftPrediction] = []
        for mc_id in split_mc_ids:
            mc_payload = self.micro_map.get(mc_id)
            if mc_payload is None:
                continue
            mc_title = str(mc_payload["mcTitle"])
            drafts.append(
                DraftPrediction(
                    mcId=mc_id,
                    mcTitle=mc_title,
                    text=generate_draft_text(mc_title, []),
                )
            )
        return [draft.__dict__ for draft in drafts]

    def predict_dataframe(self, df: pd.DataFrame) -> pd.DataFrame:
        texts = self._build_texts(df)
        x_word = self.artifacts.word_vectorizer.transform(texts)
        x_char = self.artifacts.char_vectorizer.transform(texts)
        x_matrix = hstack([x_word, x_char])
        detected_mc_ids = self._predict_detected_mc_ids(x_matrix)
        should_split_scores = self.artifacts.should_split_model.predict_proba(x_matrix)[:, 1]

        records: list[dict[str, object]] = []
        for row, item_detected_mc_ids, should_split_score in zip(
            df.itertuples(index=False),
            detected_mc_ids,
            should_split_scores,
        ):
            predicted_should_split = should_split_score >= self.artifacts.should_split_threshold
            filtered_split_mc_ids = (
                sorted(mc_id for mc_id in item_detected_mc_ids if mc_id != int(row.sourceMcId))
                if predicted_should_split
                else []
            )
            drafts = self._build_drafts(filtered_split_mc_ids)

            records.append(
                {
                    "itemId": int(row.itemId),
                    "predDetectedMcIds": sorted(mc_id for mc_id in item_detected_mc_ids if mc_id != int(row.sourceMcId)),
                    "predShouldSplit": bool(filtered_split_mc_ids),
                    "predDrafts": drafts,
                }
            )

        return pd.DataFrame(records)
