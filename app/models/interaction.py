import uuid
from datetime import datetime, timezone

from sqlalchemy import JSON, Column, DateTime, Float, ForeignKey, String
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.orm import relationship

from db.base import Base


class Interaction(Base):
    """Interaction entity for storing conversation interactions."""

    __tablename__ = "interactions"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)

    text = Column(String, nullable=False)
    timestamp = Column(DateTime, default=lambda: datetime.now(timezone.utc))
    conversation_id = Column(
        UUID(as_uuid=True), ForeignKey("conversations.id"), nullable=True
    )

    voice_embedding = Column(JSON, nullable=True)
    speaker_id = Column(UUID(as_uuid=True), ForeignKey("persons.id"), nullable=True)

    text_embedding = Column(JSON, nullable=True)
    entities = Column(JSON, nullable=True)
    topics = Column(JSON, nullable=True)
    sentiment = Column(Float, nullable=True)

    # Foreign key to network
    network_id = Column(
        UUID(as_uuid=True), ForeignKey("mira_networks.id"), nullable=False
    )

    person = relationship("Person", back_populates="interactions")
    conversation = relationship("Conversation", back_populates="interactions")
    network = relationship("MiraNetwork", back_populates="interactions")
    actions = relationship("Action", back_populates="interaction")
