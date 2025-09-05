"""
Schemas for NetworkClient model.
"""

from datetime import datetime
from typing import Optional, Dict, Any
from pydantic import BaseModel, Field


class NetworkClientBase(BaseModel):
    """Base schema for NetworkClient."""
    client_name: Optional[str] = Field(None, min_length=1, max_length=50)
    client_type: str = Field(..., pattern="^(desktop|mobile|web)$")
    client_info: Optional[Dict[str, Any]] = None
    client_settings: Optional[Dict[str, Any]] = None


class NetworkClientCreate(NetworkClientBase):
    """Schema for creating a new network client."""
    pass


class NetworkClientUpdate(BaseModel):
    """Schema for updating a network client."""
    client_name: Optional[str] = Field(None, min_length=1, max_length=50)
    client_settings: Optional[Dict[str, Any]] = None


class NetworkClientAuth(BaseModel):
    """Schema for client authentication."""
    client_id: str
    auth_token: str


class NetworkClientInfo(BaseModel):
    """Public client information."""
    client_id: str
    client_name: Optional[str]
    client_type: str
    is_online: bool
    is_recording: bool
    last_seen: Optional[datetime]
    connection_quality: float

    class Config:
        from_attributes = True


class NetworkClient(NetworkClientInfo):
    """Complete network client schema with all fields."""
    client_info: Optional[Dict[str, Any]]
    client_settings: Dict[str, Any]
    created_at: datetime
    updated_at: datetime

    class Config:
        from_attributes = True