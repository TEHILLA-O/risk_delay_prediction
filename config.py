"""Application and training configuration."""

from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
MODELS_DIR = BASE_DIR / "models"
MODEL_PATH = MODELS_DIR / "delay_risk_xgb.joblib"

FEATURE_COLUMNS: tuple[str, ...] = (
    "team_size",
    "deadline_days",
    "budget",
    "task_complexity",
    "dependencies_count",
)

# Short labels for UI charts and tables
FEATURE_LABELS: dict[str, str] = {
    "team_size": "Team size",
    "deadline_days": "Days to deadline",
    "budget": "Budget",
    "task_complexity": "Task complexity",
    "dependencies_count": "Dependencies",
}

# Human-readable clauses when a feature increases predicted delay risk (positive SHAP toward class 1)
RISK_PHRASES_POSITIVE_SHAP: dict[str, str] = {
    "team_size": "limited team capacity",
    "deadline_days": "a tight deadline",
    "budget": "constrained budget",
    "task_complexity": "high task complexity",
    "dependencies_count": "high dependency complexity",
}
