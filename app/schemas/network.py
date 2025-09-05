"""
Schemas for MiraNetwork.
"""

from datetime import datetime
from typing import Optional, Dict, Any
from pydantic import BaseModel, Field, SecretStr, UUID4


class NetworkCreate(BaseModel):
    """Schema for creating a new network."""
    name: str = Field(..., min_length=3, max_length=50)
    password: SecretStr = Field(..., min_length=8)


class NetworkAuth(BaseModel):
    """Schema for network authentication."""
    id: UUID4
    password: SecretStr


class NetworkUpdate(BaseModel):
    """Schema for updating a network."""
    name: Optional[str] = Field(None, min_length=3, max_length=50)
    service_enabled: Optional[bool] = None
    network_settings: Optional[Dict[str, Any]] = None  # Changed from settings to network_settings


class NetworkPasswordUpdate(BaseModel):
    """Schema for updating network password."""
    current_password: SecretStr
    new_password: SecretStr = Field(..., min_length=8)


class Network(BaseModel):
    """Complete network schema."""
    name: str
    id: UUID4
    service_enabled: bool
    network_settings: Dict[str, Any]
    created_at: datetime
    updated_at: datetime

    class Config:
        from_attributes = True