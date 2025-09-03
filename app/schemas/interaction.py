"""
Pydantic schemas for Interaction model.
"""

from typing import Optional, List, Any, Dict
from datetime import datetime
from pydantic import BaseModel, Field
import uuid


class InteractionBase(BaseModel):
    """Base Interaction schema."""

    text: str
    conversation_id: Optional[uuid.UUID] = None
    speaker_id: Optional[uuid.UUID] = None


class InteractionCreate(InteractionBase):
    """Schema for creating an Interaction."""

    pass


class InteractionUpdate(BaseModel):
    """Schema for updating an Interaction."""

    text: Optional[str] = None
    entities: Optional[List[Dict[str, Any]]] = None
    topics: Optional[List[str]] = None
    sentiment: Optional[float] = None


class InteractionInDB(InteractionBase):
    """Interaction schema as stored in database."""

    id: uuid.UUID
    timestamp: datetime
    voice_embedding: Optional[List[float]] = None
    text_embedding: Optional[List[float]] = None
    entities: Optional[List[Dict[str, Any]]] = None
    topics: Optional[List[str]] = None
    sentiment: Optional[float] = None

    class Config:
        from_attributes = True


class Interaction(InteractionInDB):
    """Interaction schema for API responses."""

    pass


class InteractionWithPerson(Interaction):
    """Interaction schema with person information."""

    person: Optional["Person"] = None


class InteractionWithConversation(Interaction):
    """Interaction schema with conversation information."""

    conversation: Optional["Conversation"] = None
