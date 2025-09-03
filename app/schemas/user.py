"""
Pydantic schemas for User model.
"""

from typing import Optional
from datetime import datetime
from pydantic import BaseModel
from pydantic import EmailStr
import uuid


class UserBase(BaseModel):
    """Base User schema."""

    username: str
    email: EmailStr


class UserCreate(UserBase):
    """Schema for creating a User."""

    password: str


class UserUpdate(BaseModel):
    """Schema for updating a User."""

    username: Optional[str] = None
    email: Optional[EmailStr] = None
    password: Optional[str] = None
    is_active: Optional[bool] = None


class UserInDB(UserBase):
    """User schema as stored in database."""

    id: uuid.UUID
    hashed_password: str
    is_active: bool = True
    is_superuser: bool = False
    created_at: datetime
    updated_at: datetime

    class Config:
        from_attributes = True


class User(UserInDB):
    """User schema for API responses."""

    pass


class UserLogin(BaseModel):
    """Schema for user login."""

    username: str
    password: str


class Token(BaseModel):
    """Schema for authentication token."""

    access_token: str
    token_type: str


class TokenData(BaseModel):
    """Schema for token data."""

    username: Optional[str] = None
