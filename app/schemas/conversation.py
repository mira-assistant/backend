"""
Pydantic schemas for Conversation model.
"""
from typing import Optional, List
from pydantic import BaseModel, Field
import uuid


class ConversationBase(BaseModel):
    """Base Conversation schema."""
    user_ids: Optional[List[str]] = Field(default_factory=list)


class ConversationCreate(ConversationBase):
    """Schema for creating a Conversation."""
    pass


class ConversationUpdate(BaseModel):
    """Schema for updating a Conversation."""
    topic_summary: Optional[str] = None
    context_summary: Optional[str] = None


class ConversationInDB(ConversationBase):
    """Conversation schema as stored in database."""
    id: uuid.UUID
    topic_summary: Optional[str] = None
    context_summary: Optional[str] = None

    class Config:
        from_attributes = True


class Conversation(ConversationInDB):
    """Conversation schema for API responses."""
    pass


class ConversationWithInteractions(Conversation):
    """Conversation schema with interactions."""
    interactions: List["Interaction"] = []

