"""Authentication dependencies and middleware."""

from typing import Optional
from uuid import UUID

from fastapi import Depends, HTTPException, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from sqlalchemy.orm import Session

import models
from core.security import verify_token
from db.session import get_db
from services.auth import AuthService

# Security scheme for Bearer token
security = HTTPBearer()


async def get_current_user(
    credentials: HTTPAuthorizationCredentials = Depends(security),
    db: Session = Depends(get_db)
) -> models.User:
    """Get current authenticated user from JWT token."""
    credentials_exception = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Could not validate credentials",
        headers={"WWW-Authenticate": "Bearer"},
    )

    # Verify token
    token_data = verify_token(credentials.credentials, token_type="access")
    if token_data is None:
        raise credentials_exception

    # Get user ID from token
    user_id_str = token_data.get("sub")
    if user_id_str is None:
        raise credentials_exception

    try:
        user_id = UUID(user_id_str)
    except ValueError:
        raise credentials_exception

    # Get user from database
    user = AuthService.get_user_by_id(db, user_id)
    if user is None:
        raise credentials_exception

    # Check if user is active
    if not user.is_active:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Inactive user"
        )

    return user


async def get_current_active_user(
    current_user: models.User = Depends(get_current_user)
) -> models.User:
    """Get current active user."""
    if not current_user.is_active:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Inactive user"
        )
    return current_user


def get_optional_current_user(
    credentials: Optional[HTTPAuthorizationCredentials] = Depends(HTTPBearer(auto_error=False)),
    db: Session = Depends(get_db)
) -> Optional[models.User]:
    """Get current user if token is provided, otherwise return None."""
    if credentials is None:
        return None

    try:
        # Verify token
        token_data = verify_token(credentials.credentials, token_type="access")
        if token_data is None:
            return None

        # Get user ID from token
        user_id_str = token_data.get("sub")
        if user_id_str is None:
            return None

        user_id = UUID(user_id_str)
        user = AuthService.get_user_by_id(db, user_id)
        
        # Return user only if active
        if user and user.is_active:
            return user
        return None
    except Exception:
        return None