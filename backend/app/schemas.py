from datetime import datetime
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


class PredictionRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    vessel_density: int = Field(ge=0, le=100_000)
    anchorage_vessels: int = Field(ge=0, le=100_000)
    anchorage_dwell_hours: float = Field(ge=0, le=24 * 365)
    day_of_week: int = Field(ge=0, le=6)
    month: int = Field(ge=1, le=12)
    queue_length_vessels: int | None = Field(default=None, ge=0, le=100_000)
    port_capacity_berths: int | None = Field(default=None, ge=0, le=100_000)
    wind_speed_knots: float | None = Field(default=None, ge=0, le=200)
    rainfall_mm: float | None = Field(default=None, ge=0, le=500)
    historical_delay_minutes: float | None = Field(default=None, ge=0, le=24 * 60 * 30)

    @model_validator(mode="after")
    def anchorage_count_does_not_exceed_density(self) -> "PredictionRequest":
        if self.anchorage_vessels > self.vessel_density:
            raise ValueError("anchorage_vessels cannot exceed vessel_density")
        return self


class FeatureExplanation(BaseModel):
    feature: str
    observed_value: float
    reference_value: float
    probability_change: float
    direction: Literal["increases", "decreases", "no_change"]
    method: str


class PredictionResponse(BaseModel):
    target: str
    target_definition: str
    target_status: str | None = None
    target_source: str | None = None
    prediction_horizon: str
    congestion_probability: float
    risk_category: Literal["low", "moderate", "high"]
    classification_threshold: float
    probability_calibration: str
    prediction_reliability: Literal[
        "high", "moderate", "low", "insufficient_evidence"
    ]
    reliability_reasons: list[str]
    local_explanations: list[FeatureExplanation]
    recommended_actions: list[str]
    data_quality_warnings: list[str]
    model_version: str
    prediction_timestamp: datetime
    shipment_delay_probability: float | None = None
    delay_probability: float | None = None
    delay_risk_category: Literal["low", "medium", "high"] | None = None
    estimated_delay_minutes: float | None = None
    delay_duration_minutes: float | None = None
    delay_risk_explanation: str | None = None
    delay_explanations: list[FeatureExplanation] | None = None


class DelayPredictionRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    vessel_arrivals_daily: float = Field(ge=0, le=100_000)
    queue_length_vessels: float = Field(ge=0, le=100_000)
    port_capacity_berths: float = Field(ge=1, le=100_000)
    weather_condition: str = Field(min_length=1, max_length=80)
    wind_speed_kmh: float = Field(ge=0, le=500)
    day_of_week: int = Field(ge=0, le=6)
    month: int = Field(ge=1, le=12)

    @field_validator("weather_condition")
    @classmethod
    def weather_condition_is_not_blank(cls, value: str) -> str:
        normalized = value.strip()
        if not normalized:
            raise ValueError("weather_condition must not be blank")
        return normalized


class DelayFactor(BaseModel):
    feature: str
    observed_value: float | str
    reference_value: float | str
    probability_change: float
    direction: Literal["increases", "decreases", "no_change"]
    method: str


class DelayPredictionResponse(BaseModel):
    status: Literal["available", "unavailable"]
    model_availability: Literal["available", "unavailable"]
    reason: str | None = None
    evaluation_warning: str | None = None
    predicted_outcome: Literal["delayed", "not_delayed"] | None = None
    delay_probability_percent: float | None = Field(default=None, ge=0, le=100)
    risk_category: Literal["Low", "Medium", "High"] | None = None
    estimated_delay_minutes: float | None = None
    duration_status: str
    probability_calibration: str | None = None
    probability_note: str | None = None
    risk_thresholds: dict[str, float] | None = None
    main_factors: list[DelayFactor]
    recommendations: list[str]
    target_definition: str
    dataset_provenance: dict[str, Any]
    model_version: str | None = None


class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    delay_model_loaded: bool
    data_available: bool
