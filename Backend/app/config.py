from pydantic import field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    # Use postgresql+psycopg:// — binary wheels on Windows; asyncpg not installed
    database_url: str = "postgresql+psycopg://postgres:password@localhost:5432/logistics_db"

    @field_validator("database_url", mode="before")
    @classmethod
    def normalize_database_url(cls, v: str) -> str:
        if not isinstance(v, str):
            return v
        # asyncpg → psycopg (we don't install asyncpg)
        if v.startswith("postgresql+asyncpg://"):
            return "postgresql+psycopg://" + v.split("://", 1)[1]
        # Render/Heroku often set postgres:// or postgresql:// — async SQLAlchemy needs +psycopg
        if v.startswith("postgres://"):
            return "postgresql+psycopg://" + v[len("postgres://") :]
        if v.startswith("postgresql://") and not v.startswith("postgresql+psycopg://"):
            return "postgresql+psycopg://" + v[len("postgresql://") :]
        return v
    max_upload_size_mb: int = 10
    log_level: str = "INFO"

    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8")


settings = Settings()
