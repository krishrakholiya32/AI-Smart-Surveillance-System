from datetime import datetime, timedelta, timezone
from typing import Optional

import jwt
from pwdlib import PasswordHash
from pwdlib.hashers.argon2 import Argon2Hasher
from pwdlib.hashers.bcrypt import BcryptHasher

from app.core.config import settings

# Argon2 is the current algorithm for all new/updated hashes. BcryptHasher stays
# registered only so the pre-migration admin row (and any account hashed before
# this switch) can still verify — see verify_and_rehash() and its use in the
# login route in api/auth.py.
password_hash = PasswordHash((Argon2Hasher(), BcryptHasher()))


def hash_password(password: str) -> str:
    return password_hash.hash(password)


def verify_password(plain: str, hashed: str) -> bool:
    return password_hash.verify(plain, hashed)


def verify_and_rehash(plain: str, hashed: str) -> tuple[bool, Optional[str]]:
    """Verify a password, returning a fresh Argon2 hash to persist if the
    stored hash was produced by a non-current hasher (e.g. legacy bcrypt)."""
    return password_hash.verify_and_update(plain, hashed)


def create_access_token(subject: str) -> str:
    expire = datetime.now(timezone.utc) + timedelta(minutes=settings.ACCESS_TOKEN_EXPIRE_MINUTES)
    return jwt.encode(
        {"sub": subject, "exp": expire},
        settings.SECRET_KEY,
        algorithm=settings.ALGORITHM,
    )


def decode_token(token: str) -> Optional[str]:
    try:
        payload = jwt.decode(token, settings.SECRET_KEY, algorithms=[settings.ALGORITHM])
        return payload.get("sub")
    except jwt.InvalidTokenError:
        return None
