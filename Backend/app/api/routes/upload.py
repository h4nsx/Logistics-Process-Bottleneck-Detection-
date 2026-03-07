import logging

from fastapi import APIRouter, HTTPException, UploadFile, status

from app.config import settings
from app.database import get_transaction
from app.schemas import UploadResponse
from app.services.upload_service import process_upload

router = APIRouter()
logger = logging.getLogger(__name__)

MAX_BYTES = settings.max_upload_size_mb * 1024 * 1024


@router.post("/upload", response_model=UploadResponse, status_code=status.HTTP_200_OK)
async def upload_csv(file: UploadFile) -> UploadResponse:
    """
    Upload a CSV file containing logistics execution data.

    The system will:
    - Auto-detect column mapping
    - Validate all rows
    - Insert valid executions
    - Recompute baselines
    - Detect anomalies

    All steps run inside a single transaction; failure rolls everything back.
    """
    if not file.filename or not file.filename.lower().endswith(".csv"):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Only CSV files are accepted.",
        )

    file_bytes = await file.read()
    if len(file_bytes) > MAX_BYTES:
        raise HTTPException(
            status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
            detail=f"File exceeds {settings.max_upload_size_mb} MB limit.",
        )

    try:
        async with get_transaction() as conn:
            result = await process_upload(file_bytes, conn)
    except ValueError as exc:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc)) from exc
    except Exception as exc:
        logger.exception("Upload failed unexpectedly")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Upload failed. See server logs.",
        ) from exc

    return result
