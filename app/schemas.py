from __future__ import annotations

from pydantic import BaseModel, Field


class PredictRequest(BaseModel):
    item_id: int = Field(..., description="Unique item identifier")
    source_mc_id: int = Field(..., description="Source microcategory identifier")
    description: str = Field(..., min_length=1, description="Raw item description")


class DraftResponse(BaseModel):
    mcId: int
    mcTitle: str
    text: str


class CategoryDecisionResponse(BaseModel):
    mcId: int
    mcTitle: str
    matchedPhrases: list[str]
    splitScore: float
    decision: str
    reasons: list[str]


class PredictResponse(BaseModel):
    itemId: int
    detectedMcIds: list[int]
    shouldSplit: bool
    drafts: list[DraftResponse]
    splitScores: dict[int, float]
    categoryDecisions: list[CategoryDecisionResponse]
    modelVersion: str


class HealthResponse(BaseModel):
    status: str
    modelVersion: str


class VersionResponse(BaseModel):
    service: str
    modelVersion: str
    rulesVersion: str
