"""
Authentication schemas for request/response models.
"""

from typing import Optional

from pydantic import BaseModel, EmailStr


class UserLogin(BaseModel):
    """Schema for user login request."""

    username: str
    password: str


class UserCreate(BaseModel):
    """Schema for user creation."""

    username: Optional[str] = None
    email: EmailStr
    password: Optional[str] = None


class UserResponse(BaseModel):
    """Schema for user response."""

    id: str
    username: Optional[str] = None
    email: EmailStr
    is_active: bool


class TokenResponse(BaseModel):
    """Schema for token response."""

    access_token: str
    refresh_token: str
    token_type: str = "bearer"
    user: UserResponse


class RefreshTokenRequest(BaseModel):
    """Schema for refresh token request."""

    refresh_token: str
