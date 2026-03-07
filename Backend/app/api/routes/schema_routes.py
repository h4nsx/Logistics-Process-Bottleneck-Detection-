from fastapi import APIRouter, HTTPException, UploadFile, status

from app.schemas import SchemaSuggestRequest, SchemaSuggestResponse
from app.services.schema_matcher import suggest_mapping
from app.utils.csv_parser import extract_headers

router = APIRouter()


@router.post("/schema/suggest", response_model=SchemaSuggestResponse)
async def suggest_schema_from_file(file: UploadFile) -> SchemaSuggestResponse:
    """
    Upload a CSV file and receive automatic column mapping suggestions.

    Returns confidence scores and reasoning for each detected mapping.
    Use this before the full upload to confirm or override column assignments.
    """
    if not file.filename or not file.filename.lower().endswith(".csv"):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Only CSV files are accepted.",
        )

    file_bytes = await file.read()
    try:
        headers = extract_headers(file_bytes)
    except ValueError as exc:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc)) from exc

    return suggest_mapping(headers)


@router.post("/schema/suggest/columns", response_model=SchemaSuggestResponse)
async def suggest_schema_from_columns(body: SchemaSuggestRequest) -> SchemaSuggestResponse:
    """
    Provide a list of column names directly and receive mapping suggestions.

    Useful for frontend previews where the full file isn't needed.
    """
    if not body.columns:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="columns list must not be empty.",
        )
    return suggest_mapping(body.columns)
