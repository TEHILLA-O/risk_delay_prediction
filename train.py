"""Train XGBoost delay-risk model and persist artifact for the API."""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import classification_report, roc_auc_score
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier

from config import FEATURE_COLUMNS, MODEL_PATH

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

RANDOM_STATE = 42


def _sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


def generate_synthetic_projects(n_samples: int, seed: int) -> pd.DataFrame:
    """
    Synthetic PMO-style data: delay risk rises with tight deadlines, small teams,
    low budget, high complexity, and many dependencies.
    """
    rng = np.random.default_rng(seed)
    team_size = rng.integers(2, 25, size=n_samples).astype(float)
    deadline_days = rng.integers(5, 180, size=n_samples).astype(float)
    budget = rng.uniform(10_000, 2_000_000, size=n_samples).astype(float)
    task_complexity = rng.uniform(1, 10, size=n_samples).astype(float)
    dependencies_count = rng.integers(0, 30, size=n_samples).astype(float)

    # Logistic-style latent score -> binary label with noise
    z = (
        -0.12 * team_size
        -0.04 * deadline_days
        -1.5e-6 * budget
        +0.35 * task_complexity
        +0.08 * dependencies_count
        + rng.normal(0, 0.9, size=n_samples)
    )
    p = _sigmoid(z)
    delay_risk = (rng.random(n_samples) < p).astype(int)

    df = pd.DataFrame(
        {
            "team_size": team_size,
            "deadline_days": deadline_days,
            "budget": budget,
            "task_complexity": task_complexity,
            "dependencies_count": dependencies_count,
            "delay_risk": delay_risk,
        }
    )
    return df


def load_training_frame(csv_path: Path | None, n_samples: int, seed: int) -> pd.DataFrame:
    if csv_path is not None:
        df = pd.read_csv(csv_path)
        required = set(FEATURE_COLUMNS) | {"delay_risk"}
        missing = required - set(df.columns)
        if missing:
            raise ValueError(f"CSV missing columns: {sorted(missing)}")
        return df[list(FEATURE_COLUMNS) + ["delay_risk"]]
    logger.info("No CSV provided; generating %s synthetic rows.", n_samples)
    return generate_synthetic_projects(n_samples=n_samples, seed=seed)


def train_and_save(
    df: pd.DataFrame,
    model_path: Path,
    seed: int,
) -> None:
    X = df[list(FEATURE_COLUMNS)]
    y = df["delay_risk"].astype(int)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=seed, stratify=y if y.nunique() > 1 else None
    )

    model = XGBClassifier(
        n_estimators=200,
        max_depth=5,
        learning_rate=0.08,
        subsample=0.9,
        colsample_bytree=0.9,
        reg_lambda=1.0,
        objective="binary:logistic",
        eval_metric="logloss",
        random_state=seed,
        n_jobs=0,
    )
    model.fit(X_train, y_train)

    y_prob = model.predict_proba(X_test)[:, 1]
    y_hat = (y_prob >= 0.5).astype(int)
    logger.info("ROC-AUC (holdout): %.4f", roc_auc_score(y_test, y_prob))
    logger.info("Classification report (holdout):\n%s", classification_report(y_test, y_hat))

    # Background rows for SHAP (Explainer API avoids XGBoost 2.x TreeExplainer base_score issues)
    bg_n = min(100, len(X_train))
    shap_background = X_train.sample(n=bg_n, random_state=seed)

    model_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "model": model,
        "feature_columns": FEATURE_COLUMNS,
        "shap_background": shap_background,
    }
    joblib.dump(payload, model_path)
    logger.info("Saved model bundle to %s", model_path)


def main() -> None:
    parser = argparse.ArgumentParser(description="Train delay-risk XGBoost model")
    parser.add_argument(
        "--data",
        type=Path,
        default=None,
        help="Optional CSV with columns including features and delay_risk target",
    )
    parser.add_argument("--n-samples", type=int, default=5000, help="Synthetic rows if no CSV")
    parser.add_argument("--seed", type=int, default=RANDOM_STATE)
    parser.add_argument(
        "--output",
        type=Path,
        default=MODEL_PATH,
        help="Output joblib path",
    )
    args = parser.parse_args()

    df = load_training_frame(csv_path=args.data, n_samples=args.n_samples, seed=args.seed)
    train_and_save(df, model_path=args.output, seed=args.seed)


if __name__ == "__main__":
    main()
