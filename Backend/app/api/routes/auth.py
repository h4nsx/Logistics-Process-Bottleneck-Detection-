"""Auth routes: register and login."""
import logging

from fastapi import APIRouter, HTTPException, status
from sqlalchemy import select, text

from app.database import get_connection, get_transaction
from app.models import users
from app.schemas import TokenResponse, UserLogin, UserRegister, UserResponse
from app.services.auth_service import create_access_token, hash_password, verify_password

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/auth", tags=["Auth"])


@router.post("/register", response_model=TokenResponse, status_code=status.HTTP_201_CREATED)
async def register(body: UserRegister):
    """Register a new user account."""
    async with get_connection() as conn:
        existing = await conn.execute(
            select(users.c.id).where(users.c.email == body.email)
        )
        if existing.fetchone():
            raise HTTPException(
                status_code=status.HTTP_409_CONFLICT,
                detail="Email already registered",
            )

    hashed = hash_password(body.password)

    async with get_transaction() as conn:
        result = await conn.execute(
            users.insert().values(
                full_name=body.full_name,
                email=body.email,
                hashed_password=hashed,
                is_active=True,
            ).returning(
                users.c.id,
                users.c.full_name,
                users.c.email,
                users.c.is_active,
                users.c.created_at,
            )
        )
        row = result.fetchone()

    user_out = UserResponse(
        id=row.id,
        full_name=row.full_name,
        email=row.email,
        is_active=row.is_active,
        created_at=row.created_at,
    )
    token = create_access_token({"sub": str(row.id), "email": row.email})
    logger.info("New user registered: %s", row.email)
    return TokenResponse(access_token=token, user=user_out)


@router.post("/login", response_model=TokenResponse)
async def login(body: UserLogin):
    """Login with email and password, returns JWT token."""
    async with get_connection() as conn:
        result = await conn.execute(
            select(
                users.c.id,
                users.c.full_name,
                users.c.email,
                users.c.hashed_password,
                users.c.is_active,
                users.c.created_at,
            ).where(users.c.email == body.email.lower().strip())
        )
        row = result.fetchone()

    if not row:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid email or password",
        )

    if not verify_password(body.password, row.hashed_password):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid email or password",
        )

    if not row.is_active:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Account is disabled",
        )

    user_out = UserResponse(
        id=row.id,
        full_name=row.full_name,
        email=row.email,
        is_active=row.is_active,
        created_at=row.created_at,
    )
    token = create_access_token({"sub": str(row.id), "email": row.email})
    logger.info("User logged in: %s", row.email)
    return TokenResponse(access_token=token, user=user_out)


@router.get("/me", response_model=UserResponse)
async def get_me(token: str):
    """Get current user info from JWT token. Pass token as query param for simplicity."""
    from app.services.auth_service import decode_token

    payload = decode_token(token)
    if not payload:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid token")

    user_id = int(payload.get("sub", 0))
    async with get_connection() as conn:
        result = await conn.execute(
            select(
                users.c.id,
                users.c.full_name,
                users.c.email,
                users.c.is_active,
                users.c.created_at,
            ).where(users.c.id == user_id)
        )
        row = result.fetchone()

    if not row:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="User not found")

    return UserResponse(
        id=row.id,
        full_name=row.full_name,
        email=row.email,
        is_active=row.is_active,
        created_at=row.created_at,
    )
