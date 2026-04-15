from __future__ import annotations

from dataclasses import asdict, dataclass


@dataclass(frozen=True)
class DraftPrediction:
    mcId: int
    mcTitle: str
    text: str


@dataclass(frozen=True)
class CategoryDecision:
    mcId: int
    mcTitle: str
    matchedPhrases: list[str]
    splitScore: float
    decision: str
    reasons: list[str]


@dataclass(frozen=True)
class PredictionResult:
    itemId: int
    detectedMcIds: list[int]
    shouldSplit: bool
    drafts: list[DraftPrediction]
    splitScores: dict[int, float]
    categoryDecisions: list[CategoryDecision]

    def to_dict(self) -> dict[str, object]:
        payload = asdict(self)
        payload["drafts"] = [asdict(draft) for draft in self.drafts]
        payload["categoryDecisions"] = [asdict(decision) for decision in self.categoryDecisions]
        return payload
