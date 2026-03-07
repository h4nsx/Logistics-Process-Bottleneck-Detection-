import io
from typing import Any

import pandas as pd


REQUIRED_COLUMNS = {"process_id", "step_code", "location", "start_time", "end_time"}


def extract_headers(file_bytes: bytes) -> list[str]:
    """Return column names from the first row of a CSV."""
    df = pd.read_csv(io.BytesIO(file_bytes), nrows=0)
    return list(df.columns)


def parse_csv(file_bytes: bytes) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """
    Parse CSV bytes into a list of raw row dicts.

    Returns:
        (rows, parse_errors) — parse_errors are rows that could not be read at all.
    """
    try:
        df = pd.read_csv(
            io.BytesIO(file_bytes),
            dtype=str,
            keep_default_na=False,
        )
    except Exception as exc:
        raise ValueError(f"Cannot read CSV: {exc}") from exc

    rows = df.to_dict(orient="records")
    return rows, []
