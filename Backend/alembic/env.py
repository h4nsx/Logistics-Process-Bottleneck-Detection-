import asyncio
import os
import sys
from logging.config import fileConfig

from alembic import context
from sqlalchemy.ext.asyncio import create_async_engine

from app.config import settings
from app.models import metadata


def _database_url() -> str:
    """
    Render (and other hosts) inject DATABASE_URL at runtime.
    Prefer env directly so migrations never fall back to localhost from Settings default.
    """
    url = os.environ.get("DATABASE_URL") or settings.database_url
    if not url or "localhost" in url or "127.0.0.1" in url:
        if os.environ.get("RENDER"):
            raise RuntimeError(
                "DATABASE_URL is missing or points to localhost on Render. "
                "In the Web Service → Environment, add DATABASE_URL from your PostgreSQL "
                "(Link → Internal Database URL)."
            )
    return url


def _to_asyncpg_url(url: str) -> str:
    """Ensure URL uses postgresql+asyncpg:// for async engine."""
    if url.startswith("postgresql+psycopg://"):
        return "postgresql+asyncpg://" + url.split("://", 1)[1]
    if url.startswith("postgres://"):
        return "postgresql+asyncpg://" + url[len("postgres://"):]
    if url.startswith("postgresql://") and "+" not in url.split("://")[0]:
        return "postgresql+asyncpg://" + url[len("postgresql://"):]
    return url


config = context.config
config.set_main_option("sqlalchemy.url", _to_asyncpg_url(_database_url()))

if config.config_file_name is not None:
    fileConfig(config.config_file_name)

target_metadata = metadata


def run_migrations_offline() -> None:
    url = config.get_main_option("sqlalchemy.url")
    # Strip async driver prefix for offline SQL generation
    for prefix in ("postgresql+asyncpg://", "postgresql+psycopg://"):
        if url.startswith(prefix):
            url = "postgresql://" + url.split("://", 1)[1]
            break
    context.configure(
        url=url,
        target_metadata=target_metadata,
        literal_binds=True,
        dialect_opts={"paramstyle": "named"},
    )
    with context.begin_transaction():
        context.run_migrations()


def do_run_migrations(connection):
    context.configure(connection=connection, target_metadata=target_metadata)
    with context.begin_transaction():
        context.run_migrations()


async def run_async_migrations() -> None:
    engine = create_async_engine(_to_asyncpg_url(_database_url()))
    async with engine.connect() as connection:
        await connection.run_sync(do_run_migrations)
    await engine.dispose()


def run_migrations_online() -> None:
    if sys.platform == "win32":
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
    asyncio.run(run_async_migrations())


if context.is_offline_mode():
    run_migrations_offline()
else:
    run_migrations_online()
