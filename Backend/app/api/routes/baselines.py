from fastapi import APIRouter, Query
from sqlalchemy import text

from app.database import get_connection
from app.db import queries
from app.schemas import BaselineRecord, BaselinesResponse

router = APIRouter()


@router.get("/baselines", response_model=BaselinesResponse)
async def list_baselines(
    step_code: str | None = Query(default=None, description="Filter by step code"),
    location: str | None = Query(default=None, description="Filter by location"),
) -> BaselinesResponse:
    """
    Retrieve current statistical baselines per (step_code, location).

    Baselines are recomputed automatically after every upload.
    They define what 'normal' looks like for each process step.
    """
    async with get_connection() as conn:
        rows = (
            await conn.execute(
                text(queries.GET_BASELINES),
                {"step_code": step_code, "location": location},
            )
        ).mappings().all()

    return BaselinesResponse(
        baselines=[BaselineRecord(**dict(row)) for row in rows]
    )
