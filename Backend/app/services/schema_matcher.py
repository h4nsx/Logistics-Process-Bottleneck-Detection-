"""
Intelligent schema mapping: matches user CSV column names to the required
logistics schema using keyword + fuzzy (Levenshtein) matching.
"""
import re

import Levenshtein

from app.schemas import ColumnSuggestion, SchemaSuggestResponse

REQUIRED_FIELDS = ["process_id", "step_code", "location", "start_time", "end_time"]

FIELD_KEYWORDS: dict[str, list[str]] = {
    "process_id": ["process", "id", "order", "shipment", "tracking", "ref", "reference", "job"],
    "step_code": ["step", "code", "stage", "phase", "activity", "task", "operation"],
    "location": ["location", "site", "facility", "warehouse", "hub", "zone", "plant", "depot"],
    "start_time": ["start", "begin", "from", "departure", "pickup", "started", "open"],
    "end_time": ["end", "finish", "to", "arrival", "delivery", "complete", "closed", "done"],
}


def _normalize(name: str) -> str:
    """Lowercase and strip all non-alphanumeric characters."""
    return re.sub(r"[^a-z0-9]", "", name.lower())


def _score_column(normalized_col: str, target_field: str) -> tuple[float, str]:
    """
    Compute a confidence score [0, 100] for mapping a column to a target field.

    Returns (confidence, reasoning).
    """
    keywords = FIELD_KEYWORDS[target_field]
    normalized_target = _normalize(target_field)

    # Exact match against the field name itself
    if normalized_col == normalized_target:
        return 100.0, f"Exact match: '{target_field}'"

    # Keyword exact match
    for kw in keywords:
        if normalized_col == _normalize(kw):
            return 90.0, f"Exact keyword match: '{kw}'"

    # Partial keyword match (keyword contained in column or vice versa)
    best_partial: tuple[float, str] | None = None
    for kw in keywords:
        nkw = _normalize(kw)
        if nkw in normalized_col or normalized_col in nkw:
            score = 70.0 + 10.0 * (len(nkw) / max(len(normalized_col), 1))
            if best_partial is None or score > best_partial[0]:
                best_partial = (score, f"Partial match: '{kw}' in column name")

    if best_partial and best_partial[0] >= 70:
        return best_partial

    # Fuzzy match — Levenshtein ratio against field name and keywords
    best_fuzzy_score = 0.0
    best_fuzzy_kw = target_field
    for kw in [target_field, *keywords]:
        ratio = Levenshtein.ratio(normalized_col, _normalize(kw))
        if ratio > best_fuzzy_score:
            best_fuzzy_score = ratio
            best_fuzzy_kw = kw

    fuzzy_confidence = best_fuzzy_score * 60.0  # max 60 for fuzzy
    return fuzzy_confidence, f"Fuzzy match: '{best_fuzzy_kw}' (ratio={best_fuzzy_score:.2f})"


def suggest_mapping(columns: list[str]) -> SchemaSuggestResponse:
    """
    For each user CSV column, find the best matching required field.
    Each required field can only be claimed by one column (greedy, highest score wins).
    """
    # Compute (column, field) → (score, reasoning) matrix
    scores: dict[tuple[str, str], tuple[float, str]] = {}
    for col in columns:
        normalized = _normalize(col)
        for field in REQUIRED_FIELDS:
            score, reasoning = _score_column(normalized, field)
            scores[(col, field)] = (score, reasoning)

    # Greedy assignment: pick highest-score pairs, each field assigned once
    suggestions: dict[str, ColumnSuggestion] = {}
    assigned_fields: set[str] = set()
    assigned_columns: set[str] = set()

    # Sort all (col, field) pairs by score descending
    sorted_pairs = sorted(scores.items(), key=lambda x: x[1][0], reverse=True)

    for (col, field), (score, reasoning) in sorted_pairs:
        if col in assigned_columns or field in assigned_fields:
            continue
        if score >= 40.0:  # minimum threshold
            suggestions[col] = ColumnSuggestion(
                mapped_to=field, confidence=round(score, 1), reasoning=reasoning
            )
            assigned_fields.add(field)
            assigned_columns.add(col)

    unmapped = [col for col in columns if col not in assigned_columns]
    missing_required = [f for f in REQUIRED_FIELDS if f not in assigned_fields]

    return SchemaSuggestResponse(
        suggestions=suggestions,
        unmapped_columns=unmapped,
        missing_required=missing_required,
    )


def build_column_map(suggestions: dict[str, ColumnSuggestion]) -> dict[str, str]:
    """
    Convert suggestion dict { csv_col → ColumnSuggestion } into
    a mapping { target_field → csv_col } for use in validation/transform.
    """
    return {suggestion.mapped_to: col for col, suggestion in suggestions.items()}
