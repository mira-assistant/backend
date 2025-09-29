"""
Authentication router with login, OAuth2, and token refresh endpoints.
"""

from typing import Optional
import uuid

from fastapi import APIRouter, Depends, HTTPException, status, Request
from sqlalchemy.orm import Session
from starlette.responses import RedirectResponse

import db
import models
import schemas.auth as auth_schemas
from core.auth import verify_password, get_password_hash, create_access_token, create_refresh_token, verify_token
from core.oauth import oauth, extract_user_info_google, extract_user_info_github
from api.deps import get_current_user_required

router = APIRouter(prefix="/auth", tags=["authentication"])


@router.post("/register", response_model=auth_schemas.TokenResponse)
async def register(
    user_create: auth_schemas.UserCreate,
    db_session: Session = Depends(db.get_db),
):
    """Register a new user with username and password."""
    # Check if user already exists
    existing_user = (
        db_session.query(models.User)
        .filter(
            (models.User.email == user_create.email) |
            (models.User.username == user_create.username if user_create.username else False)
        )
        .first()
    )
    
    if existing_user:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="User with this email or username already exists",
        )
    
    if not user_create.password:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Password is required for registration",
        )
    
    # Create new user
    hashed_password = get_password_hash(user_create.password)
    user = models.User(
        username=user_create.username,
        email=user_create.email,
        hashed_password=hashed_password,
        is_active=True,
    )
    
    db_session.add(user)
    db_session.commit()
    db_session.refresh(user)
    
    # Create tokens
    access_token = create_access_token(data={"sub": str(user.id)})
    refresh_token = create_refresh_token(data={"sub": str(user.id)})
    
    return auth_schemas.TokenResponse(
        access_token=access_token,
        refresh_token=refresh_token,
        user=auth_schemas.UserResponse(
            id=str(user.id),
            username=user.username,
            email=user.email,
            is_active=user.is_active,
        )
    )


@router.post("/login", response_model=auth_schemas.TokenResponse)
async def login(
    user_login: auth_schemas.UserLogin,
    db_session: Session = Depends(db.get_db),
):
    """Login with username and password."""
    # Find user by username or email
    user = (
        db_session.query(models.User)
        .filter(
            (models.User.username == user_login.username) |
            (models.User.email == user_login.username)
        )
        .first()
    )
    
    if not user or not user.hashed_password or not verify_password(user_login.password, user.hashed_password):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect username or password",
            headers={"WWW-Authenticate": "Bearer"},
        )
    
    if not user.is_active:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="User account is disabled",
        )
    
    # Create tokens
    access_token = create_access_token(data={"sub": str(user.id)})
    refresh_token = create_refresh_token(data={"sub": str(user.id)})
    
    return auth_schemas.TokenResponse(
        access_token=access_token,
        refresh_token=refresh_token,
        user=auth_schemas.UserResponse(
            id=str(user.id),
            username=user.username,
            email=user.email,
            is_active=user.is_active,
        )
    )


@router.get("/google/login")
async def google_login(request: Request):
    """Initiate Google OAuth2 login."""
    redirect_uri = request.url_for('google_callback')
    return await oauth.google.authorize_redirect(request, redirect_uri)


@router.get("/google/callback", response_model=auth_schemas.TokenResponse)
async def google_callback(
    request: Request,
    db_session: Session = Depends(db.get_db),
):
    """Handle Google OAuth2 callback."""
    try:
        token = await oauth.google.authorize_access_token(request)
        user_info = token.get('userinfo')
        
        if not user_info:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Failed to get user information from Google"
            )
        
        google_data = extract_user_info_google(user_info)
        
        # Find or create user
        user = db_session.query(models.User).filter(models.User.google_id == google_data['google_id']).first()
        
        if not user:
            # Check if user exists with same email
            user = db_session.query(models.User).filter(models.User.email == google_data['email']).first()
            if user:
                # Link Google account to existing user
                user.google_id = google_data['google_id']
            else:
                # Create new user
                user = models.User(
                    email=google_data['email'],
                    google_id=google_data['google_id'],
                    username=google_data['username'],
                    is_active=True,
                )
                db_session.add(user)
        
        db_session.commit()
        
        # Create tokens
        access_token = create_access_token(data={"sub": str(user.id)})
        refresh_token = create_refresh_token(data={"sub": str(user.id)})
        
        return auth_schemas.TokenResponse(
            access_token=access_token,
            refresh_token=refresh_token,
            user=auth_schemas.UserResponse(
                id=str(user.id),
                username=user.username,
                email=user.email,
                is_active=user.is_active,
            )
        )
        
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"OAuth2 authentication failed: {str(e)}"
        )


@router.get("/github/login")
async def github_login(request: Request):
    """Initiate GitHub OAuth2 login."""
    redirect_uri = request.url_for('github_callback')
    return await oauth.github.authorize_redirect(request, redirect_uri)


@router.get("/github/callback", response_model=auth_schemas.TokenResponse)
async def github_callback(
    request: Request,
    db_session: Session = Depends(db.get_db),
):
    """Handle GitHub OAuth2 callback."""
    try:
        token = await oauth.github.authorize_access_token(request)
        
        # Get user info from GitHub API
        resp = await oauth.github.get('user', token=token)
        user_info = resp.json()
        
        # Get user emails
        email_resp = await oauth.github.get('user/emails', token=token)
        email_info = email_resp.json()
        
        github_data = extract_user_info_github(user_info, email_info)
        
        if not github_data['email']:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="GitHub account must have a public email address"
            )
        
        # Find or create user
        user = db_session.query(models.User).filter(models.User.github_id == github_data['github_id']).first()
        
        if not user:
            # Check if user exists with same email
            user = db_session.query(models.User).filter(models.User.email == github_data['email']).first()
            if user:
                # Link GitHub account to existing user
                user.github_id = github_data['github_id']
            else:
                # Create new user
                user = models.User(
                    email=github_data['email'],
                    github_id=github_data['github_id'],
                    username=github_data['username'],
                    is_active=True,
                )
                db_session.add(user)
        
        db_session.commit()
        
        # Create tokens
        access_token = create_access_token(data={"sub": str(user.id)})
        refresh_token = create_refresh_token(data={"sub": str(user.id)})
        
        return auth_schemas.TokenResponse(
            access_token=access_token,
            refresh_token=refresh_token,
            user=auth_schemas.UserResponse(
                id=str(user.id),
                username=user.username,
                email=user.email,
                is_active=user.is_active,
            )
        )
        
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"OAuth2 authentication failed: {str(e)}"
        )


@router.post("/refresh", response_model=auth_schemas.TokenResponse)
async def refresh_token(
    refresh_request: auth_schemas.RefreshTokenRequest,
    db_session: Session = Depends(db.get_db),
):
    """Refresh access token using refresh token."""
    payload = verify_token(refresh_request.refresh_token, token_type="refresh")
    if not payload:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or expired refresh token",
            headers={"WWW-Authenticate": "Bearer"},
        )
    
    user_id = payload.get("sub")
    if not user_id:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid token payload",
        )
    
    user = db_session.query(models.User).filter(models.User.id == uuid.UUID(user_id)).first()
    if not user or not user.is_active:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="User not found or inactive",
        )
    
    # Create new tokens
    access_token = create_access_token(data={"sub": str(user.id)})
    new_refresh_token = create_refresh_token(data={"sub": str(user.id)})
    
    return auth_schemas.TokenResponse(
        access_token=access_token,
        refresh_token=new_refresh_token,
        user=auth_schemas.UserResponse(
            id=str(user.id),
            username=user.username,
            email=user.email,
            is_active=user.is_active,
        )
    )