"""
Pydantic schemas for Action model.
"""

from typing import Optional
from datetime import datetime
from pydantic import BaseModel, Field
import uuid


class ActionBase(BaseModel):
    """Base Action schema."""

    user_id: uuid.UUID
    person_id: Optional[uuid.UUID] = None
    action_type: str
    details: Optional[str] = None
    interaction_id: Optional[uuid.UUID] = None
    conversation_id: Optional[uuid.UUID] = None


class ActionCreate(ActionBase):
    """Schema for creating an Action."""

    pass


class ActionUpdate(BaseModel):
    """Schema for updating an Action."""

    status: Optional[str] = None
    scheduled_time: Optional[datetime] = None
    completed_time: Optional[datetime] = None
    details: Optional[str] = None


class ActionInDB(ActionBase):
    """Action schema as stored in database."""

    id: uuid.UUID
    status: str = "pending"
    scheduled_time: Optional[datetime] = None
    completed_time: Optional[datetime] = None

    class Config:
        from_attributes = True


class Action(ActionInDB):
    """Action schema for API responses."""

    pass
