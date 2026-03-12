from datetime import datetime
from typing import Any

from pydantic import BaseModel, ConfigDict, field_validator


# ── Upload ────────────────────────────────────────────────────────────────────

class ValidationError(BaseModel):
    row: int
    error: str


class UploadResponse(BaseModel):
    status: str
    processed_rows: int
    invalid_rows: int
    baselines_updated: bool
    anomalies_detected: int
    processing_time_seconds: float
    validation_errors: list[ValidationError] = []


# ── Anomalies ─────────────────────────────────────────────────────────────────

class AnomalyRecord(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: int
    process_id: str
    step_code: str
    location: str
    duration_minutes: float
    z_score: float | None
    risk_percent: float
    detected_at: datetime
    severity: str = ""

    @field_validator("severity", mode="before")
    @classmethod
    def compute_severity(cls, v: Any, info: Any) -> str:
        risk = float(info.data.get("risk_percent") or 0)
        if risk >= 100:
            return "High Risk"
        if risk >= 80:
            return "Warning"
        return "Normal"


class AnomaliesResponse(BaseModel):
    anomalies: list[AnomalyRecord]
    total_count: int


# ── Baselines ─────────────────────────────────────────────────────────────────

class BaselineRecord(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    step_code: str
    location: str
    mean: float
    std: float
    p95: float
    sample_size: int
    updated_at: datetime


class BaselinesResponse(BaseModel):
    baselines: list[BaselineRecord]


# ── Process Detail ────────────────────────────────────────────────────────────

class StepDetail(BaseModel):
    step_code: str
    location: str
    start_time: datetime
    end_time: datetime
    duration_minutes: float
    is_anomaly: bool
    risk_percent: float | None
    z_score: float | None
    baseline_mean: float | None
    baseline_p95: float | None


class ProcessDetailResponse(BaseModel):
    process_id: str
    steps: list[StepDetail]
    total_duration_minutes: float
    anomaly_count: int


# ── Schema Suggestion ─────────────────────────────────────────────────────────

class ColumnSuggestion(BaseModel):
    mapped_to: str
    confidence: float
    reasoning: str


class SchemaSuggestRequest(BaseModel):
    columns: list[str]


class SchemaSuggestResponse(BaseModel):
    suggestions: dict[str, ColumnSuggestion]
    unmapped_columns: list[str]
    missing_required: list[str]


# ── ML Predictions ─────────────────────────────────────────────────────────────

class StepAnomalyDetail(BaseModel):
    step_code: str
    duration_min: float
    baseline_mean: float
    baseline_p95: float
    z_score: float | None
    exceeds_p95: bool


class MLCasePrediction(BaseModel):
    case_id: str
    process_code: str
    anomaly_score: float
    risk_percentile: float
    is_anomaly: bool
    bottleneck_steps: list[StepAnomalyDetail]
    total_duration_min: float
    step_count: int

    @property
    def risk_label(self) -> str:
        if self.risk_percentile >= 80:
            return "High Risk"
        if self.risk_percentile >= 50:
            return "Warning"
        return "Normal"


class MLAnalyzeResponse(BaseModel):
    status: str
    total_cases: int
    anomaly_count: int
    process_breakdown: dict[str, int]
    predictions: list[MLCasePrediction]
    skipped_cases: int
    processing_time_seconds: float


class MLPredictionRecord(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: int
    case_id: str
    process_code: str
    anomaly_score: float
    risk_percentile: float
    is_anomaly: bool
    bottleneck_steps: list[StepAnomalyDetail]
    total_duration_min: float
    step_count: int
    analyzed_at: datetime


class MLPredictionsResponse(BaseModel):
    predictions: list[MLPredictionRecord]
    total_count: int


class MLStatusResponse(BaseModel):
    loaded_models: list[str]
    supported_processes: list[str]
    model_base_dir: str
