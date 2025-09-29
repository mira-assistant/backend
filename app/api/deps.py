"""
Authentication middleware and dependencies.
"""

from typing import Optional, Union
import uuid

from fastapi import Depends, HTTPException, status, Request
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from sqlalchemy.orm import Session

import db
import models
from core.auth import verify_token

security = HTTPBearer(auto_error=False)


async def get_current_user_optional(
    credentials: Optional[HTTPAuthorizationCredentials] = Depends(security),
    db_session: Session = Depends(db.get_db),
) -> Optional[models.User]:
    """
    Get current user from JWT token if present, otherwise return None.
    This allows endpoints to work with or without authentication.
    """
    if not credentials:
        return None
    
    payload = verify_token(credentials.credentials)
    if not payload:
        return None
    
    user_id = payload.get("sub")
    if not user_id:
        return None
    
    user = db_session.query(models.User).filter(models.User.id == uuid.UUID(user_id)).first()
    return user


async def get_current_user_required(
    credentials: HTTPAuthorizationCredentials = Depends(security),
    db_session: Session = Depends(db.get_db),
) -> models.User:
    """
    Get current user from JWT token (required).
    Raises HTTPException if token is missing or invalid.
    """
    if not credentials:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Authorization header missing",
            headers={"WWW-Authenticate": "Bearer"},
        )
    
    payload = verify_token(credentials.credentials)
    if not payload:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or expired token",
            headers={"WWW-Authenticate": "Bearer"},
        )
    
    user_id = payload.get("sub")
    if not user_id:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid token payload",
            headers={"WWW-Authenticate": "Bearer"},
        )
    
    user = db_session.query(models.User).filter(models.User.id == uuid.UUID(user_id)).first()
    if not user:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="User not found",
            headers={"WWW-Authenticate": "Bearer"},
        )
    
    if not user.is_active:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="User account is disabled",
            headers={"WWW-Authenticate": "Bearer"},
        )
    
    return user


def get_user_or_network_id(
    current_user: Optional[models.User] = Depends(get_current_user_optional),
    request: Request = None,
) -> Union[str, None]:
    """
    Get user ID if authenticated, otherwise fallback to network_id from path.
    This enables backward compatibility with existing endpoints.
    """
    if current_user:
        return str(current_user.id)
    
    # Fallback to network_id from path parameters
    if request and hasattr(request, "path_params"):
        return request.path_params.get("network_id")
    
    return None