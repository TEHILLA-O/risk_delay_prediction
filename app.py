"""FastAPI application: delay risk prediction and SHAP explanations."""

from __future__ import annotations

import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from config import BASE_DIR, FEATURE_COLUMNS, FEATURE_LABELS
from predict import (
    build_natural_language_explanation,
    load_bundle,
    predict_delay_risk_probability,
    shap_values_as_dict,
    shap_values_for_instance,
)
from schemas import ExplainResponse, PredictResponse, ProjectFeatures

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(_app: FastAPI):
    try:
        load_bundle()
        logger.info("Model loaded from disk.")
        # First SHAP call initializes internal state; warm up so the first API request stays fast.
        warmup = {name: 1.0 for name in FEATURE_COLUMNS}
        shap_values_for_instance(warmup)
        logger.info("SHAP explainer warmed up.")
    except FileNotFoundError as e:
        logger.warning("%s — prediction endpoints will fail until you run train.py", e)
    except Exception:
        logger.exception("Model or SHAP warm-up failed; /explain may be slow on first call.")
    yield


app = FastAPI(
    title="PMO Delay Risk API",
    version="1.0.0",
    description="XGBoost-based project delay risk scoring with SHAP explanations.",
    lifespan=lifespan,
)

STATIC_DIR = BASE_DIR / "static"
if STATIC_DIR.is_dir():
    app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")


@app.get("/")
def ui_root() -> FileResponse:
    index = STATIC_DIR / "index.html"
    if not index.is_file():
        raise HTTPException(status_code=404, detail="UI not found (static/index.html).")
    return FileResponse(index)


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/predict", response_model=PredictResponse)
def predict(body: ProjectFeatures) -> PredictResponse:
    try:
        feats = body.model_dump()
        p = predict_delay_risk_probability(feats)
    except FileNotFoundError as e:
        raise HTTPException(status_code=503, detail=str(e)) from e
    except Exception as e:
        logger.exception("predict failed")
        raise HTTPException(status_code=400, detail=str(e)) from e
    pct = int(round(p * 100))
    return PredictResponse(delay_risk_probability=p, delay_risk_percent=pct)


@app.post("/explain", response_model=ExplainResponse)
def explain(body: ProjectFeatures) -> ExplainResponse:
    try:
        feats = body.model_dump()
        p = predict_delay_risk_probability(feats)
        phi, _ = shap_values_for_instance(feats)
        text = build_natural_language_explanation(p, phi)
    except FileNotFoundError as e:
        raise HTTPException(status_code=503, detail=str(e)) from e
    except Exception as e:
        logger.exception("explain failed")
        raise HTTPException(status_code=400, detail=str(e)) from e
    pct = int(round(p * 100))
    return ExplainResponse(
        delay_risk_probability=p,
        delay_risk_percent=pct,
        explanation=text,
        shap_values=shap_values_as_dict(phi),
        feature_labels=dict(FEATURE_LABELS),
    )
