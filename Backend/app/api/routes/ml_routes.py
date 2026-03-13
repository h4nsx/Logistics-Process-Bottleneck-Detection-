"""
ML inference endpoints.

POST /api/ml/analyze
    Upload a CSV (process_code, case_id, step_code, start_time, end_time),
    run IsolationForest scoring per case, persist results, return analysis.

GET  /api/ml/predictions
    Query stored ML predictions with optional filters.

GET  /api/ml/status
    Show which models are currently loaded.
"""
import csv
import io
import json
import logging
import time
from datetime import datetime

from fastapi import APIRouter, HTTPException, Query, UploadFile, status
from sqlalchemy import text

from app.config import settings
from app.database import get_connection, get_transaction
from app.schemas import (
    MLAnalyzeResponse,
    MLCasePrediction,
    MLPredictionRecord,
    MLPredictionsResponse,
    MLStatusResponse,
    StepAnomalyDetail,
)
from app.services import ml_service

router = APIRouter()
logger = logging.getLogger(__name__)

TIMESTAMP_FORMATS = [
    "%Y-%m-%dT%H:%M:%S",
    "%Y-%m-%dT%H:%M:%SZ",
    "%Y-%m-%d %H:%M:%S",
    "%Y-%m-%dT%H:%M:%S.%f",
    "%Y-%m-%d",
]

_INSERT_ML_PREDICTION = """
INSERT INTO ml_predictions
    (case_id, process_code, anomaly_score, risk_percentile,
     is_anomaly, bottleneck_steps, total_duration_min, step_count)
VALUES
    (:case_id, :process_code, :anomaly_score, :risk_percentile,
     :is_anomaly, :bottleneck_steps, :total_duration_min, :step_count)
ON CONFLICT DO NOTHING;
"""

_GET_PREDICTIONS = """
SELECT id, case_id, process_code, anomaly_score, risk_percentile,
       is_anomaly, bottleneck_steps, total_duration_min, step_count, analyzed_at
FROM ml_predictions
WHERE (CAST(:process_code  AS TEXT) IS NULL OR process_code = :process_code)
  AND (CAST(:only_anomalies AS TEXT) IS NULL OR is_anomaly  = :only_anomalies)
ORDER BY risk_percentile DESC, analyzed_at DESC
LIMIT :limit;
"""

_COUNT_PREDICTIONS = """
SELECT COUNT(*) FROM ml_predictions
WHERE (CAST(:process_code  AS TEXT) IS NULL OR process_code = :process_code)
  AND (CAST(:only_anomalies AS TEXT) IS NULL OR is_anomaly  = :only_anomalies);
"""


# ── Helpers ────────────────────────────────────────────────────────────────────

def _parse_ts(value: str) -> datetime | None:
    for fmt in TIMESTAMP_FORMATS:
        try:
            return datetime.strptime(value.strip(), fmt)
        except ValueError:
            continue
    return None


def _parse_events_csv(file_bytes: bytes) -> dict[tuple[str, str], dict[str, float]]:
    """
    Parse events CSV → { (case_id, process_code): { step_code: duration_min } }

    Expected columns: process_code, case_id, step_code, start_time, end_time
    Rows with parse errors are silently skipped.
    """
    text_content = file_bytes.decode("utf-8", errors="replace").lstrip("\ufeff")
    reader = csv.DictReader(io.StringIO(text_content))

    if reader.fieldnames is None:
        raise ValueError("CSV has no header row.")

    required = {"process_code", "case_id", "step_code", "start_time", "end_time"}
    missing = required - {h.strip().lower() for h in reader.fieldnames}
    if missing:
        raise ValueError(f"Missing required columns: {', '.join(sorted(missing))}")

    cases: dict[tuple[str, str], dict[str, float]] = {}

    for row in reader:
        process_code = (row.get("process_code") or "").strip()
        case_id = (row.get("case_id") or "").strip()
        step_code = (row.get("step_code") or "").strip()
        start_raw = (row.get("start_time") or "").strip()
        end_raw = (row.get("end_time") or "").strip()

        if not all([process_code, case_id, step_code, start_raw, end_raw]):
            continue

        start = _parse_ts(start_raw)
        end = _parse_ts(end_raw)
        if start is None or end is None or end <= start:
            continue

        duration_min = (end - start).total_seconds() / 60.0
        if duration_min <= 0:
            continue

        key = (case_id, process_code)
        if key not in cases:
            cases[key] = {}
        cases[key][step_code] = duration_min

    return cases


# ── Endpoints ──────────────────────────────────────────────────────────────────

@router.post("/ml/analyze", response_model=MLAnalyzeResponse)
async def ml_analyze(file: UploadFile) -> MLAnalyzeResponse:
    """
    Upload a logistics events CSV and run ML-based anomaly detection.

    **Input CSV columns**: `process_code`, `case_id`, `step_code`, `start_time`, `end_time`

    For each case the system:
    1. Computes per-step durations
    2. Builds a feature vector (step durations + aggregate stats)
    3. Scores it with the pre-trained IsolationForest model
    4. Maps the raw score to a risk percentile [0–100]
    5. Identifies individual bottleneck steps (> p95 or z-score ≥ 2)
    6. Persists the result in `ml_predictions`

    Process codes supported: `TRUCKING_DELIVERY_FLOW`, `IMPORT_CUSTOMS_CLEARANCE`,
    `WAREHOUSE_FULFILLMENT`.
    """
    if not file.filename or not file.filename.lower().endswith(".csv"):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Only CSV files are accepted.",
        )

    file_bytes = await file.read()
    t0 = time.perf_counter()

    try:
        cases = _parse_events_csv(file_bytes)
    except ValueError as exc:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc)) from exc

    if not cases:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail="No valid rows found in the CSV.",
        )

    predictions: list[MLCasePrediction] = []
    skipped = 0
    process_breakdown: dict[str, int] = {}
    rows_to_insert: list[dict] = []

    for (case_id, process_code), step_durations in cases.items():
        result = ml_service.score_case(case_id, process_code, step_durations)
        if result is None:
            skipped += 1
            continue

        bottleneck_steps = [
            StepAnomalyDetail(
                step_code=sa.step_code,
                duration_min=sa.duration_min,
                baseline_mean=sa.baseline_mean,
                baseline_p95=sa.baseline_p95,
                z_score=sa.z_score,
                exceeds_p95=sa.exceeds_p95,
            )
            for sa in result.step_anomalies
        ]

        predictions.append(
            MLCasePrediction(
                case_id=result.case_id,
                process_code=result.process_code,
                anomaly_score=result.anomaly_score,
                risk_percentile=result.risk_percentile,
                is_anomaly=result.is_anomaly,
                bottleneck_steps=bottleneck_steps,
                total_duration_min=result.total_duration_min,
                step_count=result.step_count,
            )
        )

        process_breakdown[process_code] = process_breakdown.get(process_code, 0) + 1

        rows_to_insert.append({
            "case_id": result.case_id,
            "process_code": result.process_code,
            "anomaly_score": result.anomaly_score,
            "risk_percentile": result.risk_percentile,
            "is_anomaly": "true" if result.is_anomaly else "false",
            "bottleneck_steps": json.dumps([s.model_dump() for s in bottleneck_steps]),
            "total_duration_min": result.total_duration_min,
            "step_count": result.step_count,
        })

    # Persist all predictions in one transaction
    if rows_to_insert:
        try:
            async with get_transaction() as conn:
                await conn.execute(text(_INSERT_ML_PREDICTION), rows_to_insert)
        except Exception:
            logger.exception("Failed to persist ML predictions — results still returned")

    anomaly_count = sum(1 for p in predictions if p.is_anomaly)
    elapsed = round(time.perf_counter() - t0, 3)

    logger.info(
        "ML analyze: %d cases, %d anomalies, %d skipped in %.3fs",
        len(predictions),
        anomaly_count,
        skipped,
        elapsed,
    )

    return MLAnalyzeResponse(
        status="success",
        total_cases=len(predictions),
        anomaly_count=anomaly_count,
        process_breakdown=process_breakdown,
        predictions=sorted(predictions, key=lambda p: p.risk_percentile, reverse=True),
        skipped_cases=skipped,
        processing_time_seconds=elapsed,
    )


@router.get("/ml/predictions", response_model=MLPredictionsResponse)
async def list_ml_predictions(
    process_code: str | None = Query(default=None, description="Filter by process code"),
    only_anomalies: bool | None = Query(default=None, description="True = anomalies only"),
    limit: int = Query(default=100, ge=1, le=1000),
) -> MLPredictionsResponse:
    """
    Retrieve stored ML predictions, ordered by risk (highest first).
    """
    only_anomalies_str: str | None = None
    if only_anomalies is not None:
        only_anomalies_str = "true" if only_anomalies else "false"

    params = {
        "process_code": process_code,
        "only_anomalies": only_anomalies_str,
        "limit": limit,
    }

    async with get_connection() as conn:
        rows = (await conn.execute(text(_GET_PREDICTIONS), params)).mappings().all()
        total = (await conn.execute(text(_COUNT_PREDICTIONS), params)).scalar_one()

    records: list[MLPredictionRecord] = []
    for row in rows:
        bottleneck_steps: list[StepAnomalyDetail] = []
        try:
            raw_steps = json.loads(row["bottleneck_steps"] or "[]")
            bottleneck_steps = [StepAnomalyDetail(**s) for s in raw_steps]
        except Exception:
            pass

        records.append(
            MLPredictionRecord(
                id=row["id"],
                case_id=row["case_id"],
                process_code=row["process_code"],
                anomaly_score=float(row["anomaly_score"]),
                risk_percentile=float(row["risk_percentile"]),
                is_anomaly=row["is_anomaly"] == "true",
                bottleneck_steps=bottleneck_steps,
                total_duration_min=float(row["total_duration_min"]),
                step_count=int(row["step_count"]),
                analyzed_at=row["analyzed_at"],
            )
        )

    return MLPredictionsResponse(predictions=records, total_count=total)


@router.get("/ml/status", response_model=MLStatusResponse)
async def ml_status() -> MLStatusResponse:
    """
    Show which IsolationForest models are currently loaded and ready.
    """
    store = ml_service.get_store()
    loaded = store.loaded_processes if store else []

    return MLStatusResponse(
        loaded_models=loaded,
        supported_processes=ml_service.SUPPORTED_PROCESSES,
        model_base_dir=settings.ml_model_dir,
    )
