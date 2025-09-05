"""
Interaction model for managing user interactions.
"""

from sqlalchemy import Column, Float, DateTime, ForeignKey, JSON, Text
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.orm import relationship
from datetime import datetime, timezone
import uuid

from app.db.base import Base


class Interaction(Base):
    """Interaction entity for storing conversation interactions."""

    __tablename__ = "interactions"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)

    text = Column(Text, nullable=False)
    timestamp = Column(DateTime, default=lambda: datetime.now(timezone.utc))

    voice_embedding = Column(JSON, nullable=True)
    speaker_id = Column(UUID(as_uuid=True), ForeignKey("persons.id"), nullable=True)

    text_embedding = Column(JSON, nullable=True)
    entities = Column(JSON, nullable=True)
    topics = Column(JSON, nullable=True)
    sentiment = Column(Float, nullable=True)

    # Relationships
    person = relationship("Person", back_populates="interactions")
    conversation = relationship("Conversation", back_populates="interactions")
    network = relationship("MiraNetwork", back_populates="interactions")