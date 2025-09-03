"""
Interaction model for storing conversation interactions.
"""
from sqlalchemy import Column, String, DateTime, Text, ForeignKey, JSON, Float
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.orm import relationship
from datetime import datetime, timezone
import uuid
from typing import TYPE_CHECKING

from app.db.base import Base

if TYPE_CHECKING:
    from app.models.person import Person
    from app.models.conversation import Conversation


class Interaction(Base):
    """Interaction entity for storing conversation interactions."""

    __tablename__ = "interactions"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)

    text = Column(String, nullable=False)
    timestamp = Column(DateTime, default=lambda: datetime.now(timezone.utc))
    conversation_id = Column(UUID(as_uuid=True), ForeignKey("conversations.id"), nullable=True)

    voice_embedding = Column(JSON, nullable=True)
    speaker_id = Column(UUID(as_uuid=True), ForeignKey("persons.id"), nullable=True)

    text_embedding = Column(JSON, nullable=True)
    entities = Column(JSON, nullable=True)
    topics = Column(JSON, nullable=True)
    sentiment = Column(Float, nullable=True)

    person = relationship("Person", back_populates="interactions")
    conversation = relationship("Conversation", back_populates="interactions")

