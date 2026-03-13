"""
Orchestrates the full upload pipeline inside a single database transaction.

Flow:
  parse CSV → suggest schema → transform → validate →
  insert processes → insert step_executions →
  recompute baselines → clear + detect anomalies
"""
import logging
import time
from typing import Any

from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncConnection

from app.db import queries
from app.schemas import UploadResponse
from app.services.data_transformer import apply_mapping
from app.services.schema_matcher import build_column_map, suggest_mapping
from app.utils.csv_parser import extract_headers, parse_csv
from app.utils.validator import validate_rows

logger = logging.getLogger(__name__)

_BATCH_SIZE = 500


async def _insert_processes(conn: AsyncConnection, process_ids: list[str]) -> None:
    await conn.execute(
        text(queries.INSERT_PROCESS),
        [{"process_id": pid} for pid in process_ids],
    )


async def _insert_step_executions(
    conn: AsyncConnection, rows: list[dict[str, Any]]
) -> None:
    # Insert in batches to avoid parameter limit issues
    for i in range(0, len(rows), _BATCH_SIZE):
        batch = rows[i : i + _BATCH_SIZE]
        await conn.execute(text(queries.INSERT_STEP_EXECUTION), batch)


async def _recompute_baselines(conn: AsyncConnection) -> None:
    await conn.execute(text(queries.RECOMPUTE_BASELINES))


async def _detect_anomalies(conn: AsyncConnection) -> int:
    await conn.execute(text(queries.CLEAR_ANOMALIES))
    result = await conn.execute(text(queries.DETECT_ANOMALIES))
    return result.rowcount


async def process_upload(
    file_bytes: bytes,
    conn: AsyncConnection,
) -> UploadResponse:
    t0 = time.perf_counter()

    # 1. Parse CSV
    try:
        headers = extract_headers(file_bytes)
    except ValueError as exc:
        raise ValueError(str(exc)) from exc

    raw_rows, _ = parse_csv(file_bytes)

    # 2. Auto-detect column mapping
    suggestion_response = suggest_mapping(headers)
    if suggestion_response.missing_required:
        raise ValueError(
            f"Missing required columns: {', '.join(suggestion_response.missing_required)}"
        )

    # 3. Transform column names to canonical schema
    transformed = apply_mapping(raw_rows, suggestion_response.suggestions)

    # 4. Validate rows (identity map: fields already renamed)
    valid_rows, errors = validate_rows(
        transformed,
        column_map={field: field for field in build_column_map(suggestion_response.suggestions)},
    )

    if not valid_rows:
        raise ValueError("No valid rows found after validation")

    logger.info(
        "Upload: %d valid rows, %d invalid rows",
        len(valid_rows),
        len(errors),
    )

    # 5. Persist — all inside the caller-owned transaction
    unique_pids = list({r["process_id"] for r in valid_rows})
    await _insert_processes(conn, unique_pids)
    await _insert_step_executions(conn, valid_rows)
    await _recompute_baselines(conn)
    anomaly_count = await _detect_anomalies(conn)

    elapsed = time.perf_counter() - t0
    logger.info("Upload complete in %.3fs — %d anomalies detected", elapsed, anomaly_count)

    return UploadResponse(
        status="success",
        processed_rows=len(valid_rows),
        invalid_rows=len(errors),
        baselines_updated=True,
        anomalies_detected=anomaly_count,
        processing_time_seconds=round(elapsed, 3),
        validation_errors=errors,
    )
