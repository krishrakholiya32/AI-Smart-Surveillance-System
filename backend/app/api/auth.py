from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select

from app.core.database import get_db
from app.core.security import create_access_token, verify_and_rehash
from app.models.user import User
from app.schemas.auth import LoginRequest, TokenResponse
from app.services import login_throttle

router = APIRouter(prefix="/api/auth", tags=["auth"])


@router.post("/login", response_model=TokenResponse)
async def login(payload: LoginRequest, db: AsyncSession = Depends(get_db)):
    remaining = login_throttle.seconds_until_unlocked(payload.username)
    if remaining > 0:
        raise HTTPException(
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            detail=f"Too many failed attempts. Try again in {round(remaining / 60)} minute(s).",
        )

    result = await db.execute(select(User).where(User.username == payload.username))
    user = result.scalar_one_or_none()
    ok, upgraded_hash = verify_and_rehash(payload.password, user.hashed_password) if user else (False, None)
    if not ok:
        login_throttle.record_failure(payload.username)
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid credentials")

    # Lazy migration: the pre-existing admin row (and any account hashed before
    # this switch) verifies above via the legacy bcrypt hasher still registered
    # in security.py. pwdlib hands back a fresh Argon2 hash whenever the stored
    # one wasn't made by the current hasher — persist it now, while we still
    # have the plaintext password, so it never needs bcrypt again.
    if upgraded_hash is not None:
        user.hashed_password = upgraded_hash
        await db.commit()

    login_throttle.record_success(payload.username)
    token = create_access_token(subject=str(user.id))
    return TokenResponse(access_token=token)
