"""
Schemas for interaction management.
"""

from datetime import datetime
from typing import Optional, Dict, Any
from pydantic import BaseModel, UUID4


class InteractionBase(BaseModel):
    """Base interaction schema."""
    interaction_text: str  # Changed from text to interaction_text
    interaction_data: Optional[Dict[str, Any]] = None


class InteractionCreate(InteractionBase):
    """Schema for creating a new interaction."""
    pass


class InteractionUpdate(BaseModel):
    """Schema for updating an interaction."""
    interaction_text: Optional[str] = None  # Changed from text to interaction_text
    interaction_data: Optional[Dict[str, Any]] = None


class Interaction(InteractionBase):
    """Complete interaction schema."""
    id: UUID4
    network_id: UUID4
    created_at: datetime
    updated_at: datetime

    class Config:
        from_attributes = True