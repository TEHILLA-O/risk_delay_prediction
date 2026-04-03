"""Pydantic request/response models for the API."""

from pydantic import BaseModel, Field


class ProjectFeatures(BaseModel):
    team_size: float = Field(..., ge=0, description="Number of people on the project team")
    deadline_days: float = Field(..., ge=0, description="Days until the deadline")
    budget: float = Field(..., ge=0, description="Project budget (same units as training data)")
    task_complexity: float = Field(
        ..., ge=0, description="Task complexity score (higher = more complex)"
    )
    dependencies_count: float = Field(
        ..., ge=0, description="Number of upstream/downstream dependencies"
    )


class PredictResponse(BaseModel):
    delay_risk_probability: float = Field(
        ..., ge=0, le=1, description="Predicted probability of delay (class 1)"
    )
    delay_risk_percent: int = Field(..., ge=0, le=100, description="Rounded percentage 0–100")


class ExplainResponse(BaseModel):
    delay_risk_probability: float
    delay_risk_percent: int
    explanation: str
    shap_values: dict[str, float] = Field(
        ...,
        description="SHAP contribution per feature toward predicted delay probability.",
    )
    feature_labels: dict[str, str] = Field(
        ...,
        description="Human-readable names for each model feature key.",
    )
