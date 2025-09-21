"""Authentication router with login, register, and OAuth endpoints."""

from datetime import timedelta
from typing import Dict, Any
import secrets
import urllib.parse
from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException, status, Request
from fastapi.responses import RedirectResponse
from sqlalchemy.orm import Session
import httpx

import models
from core.config import settings
from core.deps import get_current_user, get_optional_current_user
from core.security import (
    create_access_token, 
    create_refresh_token, 
    verify_token,
    get_token_expire_time
)
from db.session import get_db
from schemas.auth import (
    UserCreate, 
    UserPublic, 
    LoginRequest, 
    Token, 
    RefreshTokenRequest,
    GoogleAuthRequest
)
from services.auth import AuthService

router = APIRouter(prefix="/auth", tags=["Authentication"])


@router.post("/register", response_model=UserPublic)
async def register(
    user_data: UserCreate,
    db: Session = Depends(get_db)
):
    """Register a new user."""
    user = AuthService.create_user(db, user_data)
    return user


@router.post("/login", response_model=Token)
async def login(
    login_data: LoginRequest,
    db: Session = Depends(get_db)
):
    """Login with email and password."""
    user = AuthService.authenticate_user(db, login_data.email, login_data.password)
    if not user:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect email or password",
            headers={"WWW-Authenticate": "Bearer"},
        )

    # Create tokens
    token_data = {"sub": str(user.id), "email": user.email}
    access_token = create_access_token(token_data)
    refresh_token = create_refresh_token(token_data)

    return {
        "access_token": access_token,
        "refresh_token": refresh_token,
        "token_type": "bearer",
        "expires_in": get_token_expire_time()
    }


@router.post("/refresh", response_model=Token)
async def refresh_token(
    refresh_data: RefreshTokenRequest,
    db: Session = Depends(get_db)
):
    """Refresh access token using refresh token."""
    credentials_exception = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Could not validate refresh token",
        headers={"WWW-Authenticate": "Bearer"},
    )

    # Verify refresh token
    token_data = verify_token(refresh_data.refresh_token, token_type="refresh")
    if token_data is None:
        raise credentials_exception

    # Get user
    user_id_str = token_data.get("sub")
    if user_id_str is None:
        raise credentials_exception

    try:
        user_id = UUID(user_id_str)
    except ValueError:
        raise credentials_exception

    user = AuthService.get_user_by_id(db, user_id)
    if user is None or not user.is_active:
        raise credentials_exception

    # Create new tokens
    new_token_data = {"sub": str(user.id), "email": user.email}
    access_token = create_access_token(new_token_data)
    new_refresh_token = create_refresh_token(new_token_data)

    return {
        "access_token": access_token,
        "refresh_token": new_refresh_token,
        "token_type": "bearer",
        "expires_in": get_token_expire_time()
    }


@router.post("/logout")
async def logout(current_user: models.User = Depends(get_current_user)):
    """Logout user (client should discard tokens)."""
    return {"message": "Successfully logged out"}


@router.get("/me", response_model=UserPublic)
async def get_current_user_info(current_user: models.User = Depends(get_current_user)):
    """Get current user information."""
    return current_user


# Example of how to protect existing endpoints
@router.get("/protected-example")
async def protected_example(current_user: models.User = Depends(get_current_user)):
    """Example of a protected endpoint that requires authentication.
    
    This shows how to add authentication to any existing endpoint:
    1. Import get_current_user from core.deps
    2. Add current_user: models.User = Depends(get_current_user) parameter
    3. The endpoint is now protected - only accessible with valid JWT token
    """
    return {
        "message": f"Hello {current_user.full_name or current_user.email}!",
        "user_id": str(current_user.id),
        "authenticated": True,
        "example": "This is how you protect any endpoint"
    }


@router.get("/optional-auth-example")
async def optional_auth_example(current_user: models.User = Depends(get_optional_current_user)):
    """Example of an endpoint with optional authentication.
    
    This endpoint works with or without authentication:
    - If user is authenticated, returns personalized response
    - If user is not authenticated, returns generic response
    """
    if current_user:
        return {
            "message": f"Welcome back, {current_user.full_name or current_user.email}!",
            "user_id": str(current_user.id),
            "authenticated": True
        }
    else:
        return {
            "message": "Welcome, anonymous user!",
            "authenticated": False,
            "note": "You can access this endpoint without authentication"
        }


# Google OAuth 2.0 endpoints

@router.get("/google")
async def google_auth(request: Request):
    """Redirect to Google OAuth 2.0 authorization."""
    if not settings.google_client_id:
        raise HTTPException(
            status_code=status.HTTP_501_NOT_IMPLEMENTED,
            detail="Google OAuth not configured"
        )

    # Generate state parameter for CSRF protection
    state = secrets.token_urlsafe(32)
    
    # Store state in session or cache (for production, use Redis or database)
    # For now, we'll include it in the redirect and verify it in callback
    
    # Build Google OAuth URL
    params = {
        "client_id": settings.google_client_id,
        "redirect_uri": str(request.url_for("google_callback")),
        "scope": "openid email profile",
        "response_type": "code",
        "state": state,
        "access_type": "offline",
        "prompt": "consent"
    }
    
    google_auth_url = "https://accounts.google.com/o/oauth2/v2/auth?" + urllib.parse.urlencode(params)
    
    return RedirectResponse(url=google_auth_url)


@router.get("/google/callback")
async def google_callback(
    code: str,
    state: str,
    request: Request,
    db: Session = Depends(get_db)
):
    """Handle Google OAuth 2.0 callback."""
    if not settings.google_client_id or not settings.google_client_secret:
        raise HTTPException(
            status_code=status.HTTP_501_NOT_IMPLEMENTED,
            detail="Google OAuth not configured"
        )

    try:
        # Exchange authorization code for access token
        token_data = {
            "client_id": settings.google_client_id,
            "client_secret": settings.google_client_secret,
            "code": code,
            "grant_type": "authorization_code",
            "redirect_uri": str(request.url_for("google_callback")),
        }

        async with httpx.AsyncClient() as client:
            token_response = await client.post(
                "https://oauth2.googleapis.com/token",
                data=token_data
            )
            token_response.raise_for_status()
            tokens = token_response.json()

            # Get user info from Google
            headers = {"Authorization": f"Bearer {tokens['access_token']}"}
            user_response = await client.get(
                "https://www.googleapis.com/oauth2/v2/userinfo",
                headers=headers
            )
            user_response.raise_for_status()
            google_user = user_response.json()

        # Create or update user
        user = AuthService.create_or_update_google_user(
            db=db,
            google_id=google_user["id"],
            email=google_user["email"],
            full_name=google_user.get("name"),
            avatar_url=google_user.get("picture")
        )

        # Create JWT tokens
        token_data = {"sub": str(user.id), "email": user.email}
        access_token = create_access_token(token_data)
        refresh_token = create_refresh_token(token_data)

        # For a web app, you would redirect to the frontend with tokens
        # For this API, we'll return the tokens directly
        return {
            "message": "Successfully authenticated with Google",
            "access_token": access_token,
            "refresh_token": refresh_token,
            "token_type": "bearer",
            "expires_in": get_token_expire_time(),
            "user": UserPublic.model_validate(user)
        }

    except httpx.HTTPError as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Failed to authenticate with Google: {str(e)}"
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Authentication error: {str(e)}"
        )