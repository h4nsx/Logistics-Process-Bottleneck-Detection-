"""
Apply a validated column mapping to rename CSV rows into the canonical schema,
then compute duration_minutes for each row.
"""
from typing import Any

from app.schemas import ColumnSuggestion


def apply_mapping(
    rows: list[dict[str, Any]],
    suggestions: dict[str, ColumnSuggestion],
) -> list[dict[str, Any]]:
    """
    Rename each row's keys from user CSV column names to canonical field names.

    Only canonical fields are kept; extraneous columns are dropped.
    Duration computation is delegated to the validator for correctness.
    """
    col_to_field = {col: s.mapped_to for col, s in suggestions.items()}
    transformed = []
    for row in rows:
        new_row: dict[str, Any] = {}
        for col, field in col_to_field.items():
            if col in row:
                new_row[field] = row[col]
        transformed.append(new_row)
    return transformed
