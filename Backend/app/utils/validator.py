from datetime import datetime
from typing import Any

from app.schemas import ValidationError

REQUIRED_FIELDS = ["process_id", "step_code", "location", "start_time", "end_time"]
TIMESTAMP_FORMATS = [
    "%Y-%m-%d %H:%M:%S",
    "%Y-%m-%dT%H:%M:%S",
    "%Y-%m-%dT%H:%M:%SZ",
    "%Y-%m-%dT%H:%M:%S.%f",
    "%Y-%m-%d",
]


def _parse_timestamp(value: str) -> datetime | None:
    for fmt in TIMESTAMP_FORMATS:
        try:
            return datetime.strptime(value.strip(), fmt)
        except ValueError:
            continue
    return None


def validate_row(
    row: dict[str, Any],
    row_index: int,
    column_map: dict[str, str],
) -> tuple[dict[str, Any] | None, ValidationError | None]:
    """
    Validate a single row using the resolved column mapping.

    Returns:
        (clean_row, None) on success
        (None, ValidationError) on failure
    """
    mapped: dict[str, Any] = {}

    for target_field, source_col in column_map.items():
        value = row.get(source_col, "")
        if isinstance(value, str):
            value = value.strip()
        if not value:
            return None, ValidationError(row=row_index, error=f"'{source_col}' is empty")
        mapped[target_field] = value

    start = _parse_timestamp(str(mapped["start_time"]))
    if start is None:
        return None, ValidationError(
            row=row_index, error=f"Invalid start_time format: {mapped['start_time']!r}"
        )

    end = _parse_timestamp(str(mapped["end_time"]))
    if end is None:
        return None, ValidationError(
            row=row_index, error=f"Invalid end_time format: {mapped['end_time']!r}"
        )

    if end <= start:
        return None, ValidationError(
            row=row_index, error="end_time must be after start_time"
        )

    duration_minutes = (end - start).total_seconds() / 60.0
    if duration_minutes <= 0:
        return None, ValidationError(
            row=row_index, error="Computed duration_minutes must be positive"
        )

    return {
        "process_id": str(mapped["process_id"]),
        "step_code": str(mapped["step_code"]),
        "location": str(mapped["location"]),
        "start_time": start,           # datetime object — asyncpg requires native types
        "end_time": end,
        "duration_minutes": duration_minutes,
    }, None


def validate_rows(
    rows: list[dict[str, Any]],
    column_map: dict[str, str],
) -> tuple[list[dict[str, Any]], list[ValidationError]]:
    """Validate all rows; collect errors without aborting."""
    valid: list[dict[str, Any]] = []
    errors: list[ValidationError] = []

    for idx, row in enumerate(rows, start=2):  # row 1 = header
        clean, error = validate_row(row, idx, column_map)
        if clean:
            valid.append(clean)
        else:
            errors.append(error)

    return valid, errors
