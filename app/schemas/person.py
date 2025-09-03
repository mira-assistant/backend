"""
Pydantic schemas for Person model.
"""

from typing import Optional, List, Any
from datetime import datetime
from pydantic import BaseModel, Field
import uuid


class PersonBase(BaseModel):
    """Base Person schema."""

    name: Optional[str] = None
    index: int


class PersonCreate(PersonBase):
    """Schema for creating a Person."""

    pass


class PersonUpdate(BaseModel):
    """Schema for updating a Person."""

    name: Optional[str] = None
    voice_embedding: Optional[List[float]] = None


class PersonInDB(PersonBase):
    """Person schema as stored in database."""

    id: uuid.UUID
    voice_embedding: Optional[List[float]] = None
    cluster_id: Optional[int] = None
    created_at: datetime
    updated_at: datetime

    class Config:
        from_attributes = True


class Person(PersonInDB):
    """Person schema for API responses."""

    pass
