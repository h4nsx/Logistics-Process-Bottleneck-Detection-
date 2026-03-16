import os

from pydantic import field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

# Resolve project root (two levels up from this file: app/ → Backend/ → project root)
_BACKEND_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_PROJECT_ROOT = os.path.dirname(_BACKEND_DIR)
_DEFAULT_MODEL_DIR = os.path.join(_PROJECT_ROOT, "ML", "model", "process_models")


class Settings(BaseSettings):
    # Default for local dev only. On Render set DATABASE_URL (Internal Database URL).
    database_url: str = "postgresql+asyncpg://postgres:password@localhost:5432/logistics_db"

    # JWT secret key — MUST be set in production via SECRET_KEY env var
    secret_key: str = "change-me-in-production-use-a-long-random-string"

    @field_validator("database_url", mode="before")
    @classmethod
    def normalize_database_url(cls, v: str) -> str:
        if not isinstance(v, str):
            return v
        # Normalize all variants to postgresql+asyncpg://
        if v.startswith("postgresql+psycopg://"):
            return "postgresql+asyncpg://" + v.split("://", 1)[1]
        if v.startswith("postgres://"):
            return "postgresql+asyncpg://" + v[len("postgres://"):]
        if v.startswith("postgresql://") and "+" not in v.split("://")[0]:
            return "postgresql+asyncpg://" + v[len("postgresql://"):]
        return v

    max_upload_size_mb: int = 10
    log_level: str = "INFO"

    # Path to directory containing per-process model subdirectories.
    # Override via ML_MODEL_DIR env var in production.
    ml_model_dir: str = _DEFAULT_MODEL_DIR

    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8")


settings = Settings()
