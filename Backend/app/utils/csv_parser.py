"""
CSV parsing via stdlib only — avoids pandas (no C compiler / VS Build Tools needed on Windows).
"""
import csv
import io
from typing import Any


REQUIRED_COLUMNS = {"process_id", "step_code", "location", "start_time", "end_time"}


def _decode_bytes(file_bytes: bytes) -> str:
    """Decode as UTF-8; strip BOM if present."""
    if file_bytes.startswith(b"\xef\xbb\xbf"):
        return file_bytes[3:].decode("utf-8")
    return file_bytes.decode("utf-8", errors="replace")


def extract_headers(file_bytes: bytes) -> list[str]:
    """Return column names from the first row of a CSV."""
    text = _decode_bytes(file_bytes)
    reader = csv.reader(io.StringIO(text))
    try:
        row = next(reader)
    except StopIteration:
        return []
    return [h.strip() for h in row]


def parse_csv(file_bytes: bytes) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """
    Parse CSV bytes into a list of raw row dicts.
    All values are strings; empty cells become "".
    """
    text = _decode_bytes(file_bytes)
    try:
        reader = csv.DictReader(io.StringIO(text))
        if reader.fieldnames is None:
            raise ValueError("CSV has no header row")
        # Normalize keys to strip whitespace on headers
        rows: list[dict[str, Any]] = []
        for raw in reader:
            # DictReader may use first row keys; ensure we only keep defined fields
            row = {k.strip(): (v if v is not None else "") for k, v in raw.items() if k}
            rows.append(row)
    except Exception as exc:
        raise ValueError(f"Cannot read CSV: {exc}") from exc

    return rows, []
