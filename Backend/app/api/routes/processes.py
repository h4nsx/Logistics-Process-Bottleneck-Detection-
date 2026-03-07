from fastapi import APIRouter, HTTPException, status
from sqlalchemy import text

from app.database import get_connection
from app.db import queries
from app.schemas import ProcessDetailResponse, StepDetail

router = APIRouter()


@router.get("/process/{process_id}", response_model=ProcessDetailResponse)
async def get_process(process_id: str) -> ProcessDetailResponse:
    """
    Retrieve the full step-by-step execution timeline for a process.

    Each step includes:
    - Actual duration vs baseline (mean, p95)
    - Whether it was flagged as an anomaly
    - Risk percent and z-score
    """
    async with get_connection() as conn:
        rows = (
            await conn.execute(
                text(queries.GET_PROCESS_STEPS),
                {"process_id": process_id},
            )
        ).mappings().all()

    if not rows:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Process '{process_id}' not found.",
        )

    steps = [
        StepDetail(
            step_code=row["step_code"],
            location=row["location"],
            start_time=row["start_time"],
            end_time=row["end_time"],
            duration_minutes=float(row["duration_minutes"]),
            is_anomaly=row["risk_percent"] is not None,
            risk_percent=float(row["risk_percent"]) if row["risk_percent"] is not None else None,
            z_score=float(row["z_score"]) if row["z_score"] is not None else None,
            baseline_mean=float(row["baseline_mean"]) if row["baseline_mean"] is not None else None,
            baseline_p95=float(row["baseline_p95"]) if row["baseline_p95"] is not None else None,
        )
        for row in rows
    ]

    total_duration = sum(s.duration_minutes for s in steps)
    anomaly_count = sum(1 for s in steps if s.is_anomaly)

    return ProcessDetailResponse(
        process_id=process_id,
        steps=steps,
        total_duration_minutes=total_duration,
        anomaly_count=anomaly_count,
    )
