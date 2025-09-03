"""
API dependencies for FastAPI.
"""

from typing import Generator
from fastapi import Depends, HTTPException, status
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from sqlalchemy.orm import Session

from app.db.session import get_db
from app.core.security import get_current_user
from app.core.config import settings

# Security scheme
security = HTTPBearer()


def get_current_user_dependency(
    credentials: HTTPAuthorizationCredentials = Depends(security),
) -> dict:
    """Get current user from JWT token."""
    return get_current_user(credentials.credentials)


def get_db_dependency() -> Generator[Session, None, None]:
    """Get database session dependency."""
    yield from get_db()


def get_optional_current_user(
    credentials: HTTPAuthorizationCredentials = Depends(security),
) -> dict:
    """Get current user from JWT token (optional)."""
    try:
        return get_current_user(credentials.credentials)
    except HTTPException:
        return None
