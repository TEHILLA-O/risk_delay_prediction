"""Load trained model, score projects, and build SHAP-based explanations."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

import joblib
import numpy as np
import pandas as pd
import shap
from xgboost import XGBClassifier

from config import FEATURE_COLUMNS, MODEL_PATH, RISK_PHRASES_POSITIVE_SHAP


def shap_values_as_dict(phi: np.ndarray) -> dict[str, float]:
    return {name: float(val) for name, val in zip(FEATURE_COLUMNS, phi, strict=True)}

logger = logging.getLogger(__name__)

MITIGATING_PHRASES: dict[str, str] = {
    "team_size": "stronger staffing levels",
    "deadline_days": "a more comfortable timeline",
    "budget": "greater budget headroom",
    "task_complexity": "lower task complexity",
    "dependencies_count": "fewer cross-project dependencies",
}


@dataclass
class ModelBundle:
    model: XGBClassifier
    feature_columns: tuple[str, ...]
    explainer: Any


_bundle: ModelBundle | None = None


def _build_shap_explainer(model: XGBClassifier, background: pd.DataFrame) -> Any:
    """SHAP Explainer on P(delay); uses background data (stable with XGBoost 2.x)."""

    def delay_risk_proba(X: pd.DataFrame) -> np.ndarray:
        return model.predict_proba(X)[:, 1]

    return shap.Explainer(delay_risk_proba, background)


def _explanation_to_phi(exp: Any, n_features: int) -> np.ndarray:
    arr = np.asarray(exp.values, dtype=float)
    arr = arr.reshape(-1)
    if arr.shape[0] != n_features:
        raise ValueError(f"Unexpected SHAP shape {arr.shape} for n_features={n_features}")
    return arr


def load_bundle(force_reload: bool = False) -> ModelBundle:
    global _bundle
    if _bundle is not None and not force_reload:
        return _bundle
    if not MODEL_PATH.is_file():
        raise FileNotFoundError(
            f"Model not found at {MODEL_PATH}. Run `python train.py` first."
        )
    payload = joblib.load(MODEL_PATH)
    model: XGBClassifier = payload["model"]
    feature_columns: tuple[str, ...] = tuple(payload["feature_columns"])
    if feature_columns != FEATURE_COLUMNS:
        logger.warning("Feature columns in saved model differ from config: %s", feature_columns)
    background = payload.get("shap_background")
    if background is None:
        raise ValueError(
            "Saved model is missing shap_background. Retrain with the current train.py."
        )
    explainer = _build_shap_explainer(model, background)
    _bundle = ModelBundle(model=model, feature_columns=feature_columns, explainer=explainer)
    return _bundle


def features_to_frame(features: dict[str, float]) -> pd.DataFrame:
    missing = [c for c in FEATURE_COLUMNS if c not in features]
    if missing:
        raise ValueError(f"Missing features: {missing}")
    row = {c: float(features[c]) for c in FEATURE_COLUMNS}
    return pd.DataFrame([row], columns=list(FEATURE_COLUMNS))


def predict_delay_risk_probability(features: dict[str, float]) -> float:
    bundle = load_bundle()
    X = features_to_frame(features)
    proba = bundle.model.predict_proba(X)[0, 1]
    return float(proba)


def shap_values_for_instance(features: dict[str, float]) -> tuple[np.ndarray, pd.DataFrame]:
    bundle = load_bundle()
    X = features_to_frame(features)
    exp = bundle.explainer(X)
    phi = _explanation_to_phi(exp, len(FEATURE_COLUMNS))
    return phi, X


def _ordinal_join(phrases: list[str]) -> str:
    if not phrases:
        return ""
    if len(phrases) == 1:
        return phrases[0]
    if len(phrases) == 2:
        return f"{phrases[0]} and {phrases[1]}"
    return ", ".join(phrases[:-1]) + f", and {phrases[-1]}"


def build_natural_language_explanation(
    delay_risk_probability: float,
    phi: np.ndarray,
) -> str:
    """Compose a short narrative from SHAP contributions for the positive class."""
    pct = int(round(delay_risk_probability * 100))
    names = list(FEATURE_COLUMNS)

    # Rank by contribution toward delay (positive SHAP = higher risk)
    signed = list(zip(names, phi, strict=True))
    risk_increasing = [(n, v) for n, v in signed if v > 0]
    risk_increasing.sort(key=lambda x: x[1], reverse=True)

    phrases: list[str] = []
    for name, _ in risk_increasing[:3]:
        phrases.append(RISK_PHRASES_POSITIVE_SHAP[name])

    if not phrases:
        # Emphasize mitigating factors if the model attributes low risk to specific drivers
        mitigating = [(n, v) for n, v in signed if v < 0]
        mitigating.sort(key=lambda x: x[1])
        for name, _ in mitigating[:3]:
            phrases.append(MITIGATING_PHRASES[name])
        if phrases:
            joined = _ordinal_join(phrases)
            return (
                f"This project has a {pct}% delay risk; the model highlights {joined} "
                f"as factors reducing predicted delay risk."
            )
        joined_abs = _ordinal_join(
            [names[i] for i in np.argsort(-np.abs(phi))[:3]]
        )
        return (
            f"This project has a {pct}% delay risk; the largest absolute contributions "
            f"come from {joined_abs} (see SHAP values for directionality)."
        )

    joined = _ordinal_join(phrases)
    return f"This project has a {pct}% delay risk due to {joined}."
