from fastapi import APIRouter, Query
from sqlalchemy import text

from app.database import get_connection
from app.db import queries
from app.schemas import AnomaliesResponse, AnomalyRecord

router = APIRouter()


@router.get("/anomalies", response_model=AnomaliesResponse)
async def list_anomalies(
    limit: int = Query(default=100, ge=1, le=1000, description="Maximum results"),
    min_risk: float | None = Query(default=None, ge=0, le=100, description="Minimum risk_percent"),
) -> AnomaliesResponse:
    """
    Retrieve detected bottlenecks, ordered by risk level (highest first).
    """
    async with get_connection() as conn:
        params = {"limit": limit, "min_risk": min_risk}

        rows = (await conn.execute(text(queries.GET_ANOMALIES), params)).mappings().all()
        total = (await conn.execute(text(queries.COUNT_ANOMALIES), params)).scalar_one()

    anomaly_records = []
    for row in rows:
        risk = float(row["risk_percent"])
        severity = "High Risk" if risk >= 100 else "Warning" if risk >= 80 else "Normal"
        anomaly_records.append(
            AnomalyRecord(
                id=row["id"],
                process_id=row["process_id"],
                step_code=row["step_code"],
                location=row["location"],
                duration_minutes=float(row["duration_minutes"]),
                z_score=float(row["z_score"]) if row["z_score"] is not None else None,
                risk_percent=risk,
                detected_at=row["detected_at"],
                severity=severity,
            )
        )

    return AnomaliesResponse(anomalies=anomaly_records, total_count=total)
