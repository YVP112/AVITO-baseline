from __future__ import annotations

from fastapi import FastAPI, HTTPException

from app.schemas import HealthResponse, PredictRequest, PredictResponse, VersionResponse
from app.service import get_model_version, get_predictor, get_rules_version


app = FastAPI(
    title="Avito Services Splitter API",
    version="1.0.0",
    description="Production-style inference API for split detection and draft generation.",
)


@app.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    get_predictor()
    return HealthResponse(status="ok", modelVersion=get_model_version())


@app.get("/version", response_model=VersionResponse)
def version() -> VersionResponse:
    return VersionResponse(
        service="avito-services-splitter",
        modelVersion=get_model_version(),
        rulesVersion=get_rules_version(),
    )


@app.post("/predict", response_model=PredictResponse)
def predict(payload: PredictRequest) -> PredictResponse:
    description = payload.description.strip()
    if not description:
        raise HTTPException(status_code=422, detail="description must not be empty")

    predictor = get_predictor()
    prediction = predictor.predict_item(
        item_id=payload.item_id,
        source_mc_id=payload.source_mc_id,
        description=description,
    )

    return PredictResponse(
        itemId=prediction.itemId,
        detectedMcIds=prediction.detectedMcIds,
        shouldSplit=prediction.shouldSplit,
        drafts=prediction.drafts,
        splitScores={int(key): float(value) for key, value in prediction.splitScores.items()},
        categoryDecisions=prediction.categoryDecisions,
        modelVersion=get_model_version(),
    )
