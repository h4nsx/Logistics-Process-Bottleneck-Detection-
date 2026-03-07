from contextlib import asynccontextmanager
from typing import AsyncGenerator

from sqlalchemy.ext.asyncio import AsyncConnection, AsyncEngine, create_async_engine

from app.config import settings

engine: AsyncEngine = create_async_engine(
    settings.database_url,
    pool_pre_ping=True,
    pool_size=10,
    max_overflow=20,
    echo=False,
)


@asynccontextmanager
async def get_connection() -> AsyncGenerator[AsyncConnection, None]:
    async with engine.connect() as conn:
        yield conn


@asynccontextmanager
async def get_transaction() -> AsyncGenerator[AsyncConnection, None]:
    async with engine.begin() as conn:
        yield conn


async def dispose_engine() -> None:
    await engine.dispose()
