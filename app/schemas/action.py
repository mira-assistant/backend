"""
Schemas for action/task management.
"""

from datetime import datetime
from typing import Optional, Dict, Any
from pydantic import BaseModel, UUID4


class ActionBase(BaseModel):
    """Base action schema."""
    title: str
    description: Optional[str] = None
    action_data: Optional[Dict[str, Any]] = None


class ActionCreate(ActionBase):
    """Schema for creating a new action."""
    pass


class ActionUpdate(BaseModel):
    """Schema for updating an action."""
    title: Optional[str] = None
    description: Optional[str] = None
    action_status: Optional[str] = None  # Changed from status to action_status
    is_completed: Optional[bool] = None
    action_data: Optional[Dict[str, Any]] = None


class Action(ActionBase):
    """Complete action schema."""
    id: UUID4
    network_id: UUID4
    action_status: str  # Changed from status to action_status
    is_completed: bool
    created_at: datetime
    updated_at: datetime
    completed_at: Optional[datetime] = None

    class Config:
        from_attributes = True