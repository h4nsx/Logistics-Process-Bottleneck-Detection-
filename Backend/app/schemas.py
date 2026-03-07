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
    z_score: float
    risk_percent: float
    detected_at: datetime
    severity: str = ""

    @field_validator("severity", mode="before")
    @classmethod
    def compute_severity(cls, v: Any, info: Any) -> str:
        risk = info.data.get("risk_percent", 0)
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
