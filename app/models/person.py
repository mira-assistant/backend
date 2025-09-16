import uuid
from datetime import datetime, timezone

from sqlalchemy import JSON, Column, DateTime, ForeignKey, Integer, String
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.orm import relationship

from db.base import Base

from .conversation import person_conversation_association


class Person(Base):
    """Person entity for speaker recognition and management."""

    __tablename__ = "persons"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    name = Column(String, nullable=True)
    index = Column(Integer, unique=True, nullable=False)

    voice_embedding = Column(JSON, nullable=True)
    cluster_id = Column(Integer, nullable=True)

    # Foreign key to network
    network_id = Column(
        UUID(as_uuid=True), ForeignKey("mira_networks.id"), nullable=False
    )

    created_at = Column(DateTime, default=lambda: datetime.now(timezone.utc))
    updated_at = Column(
        DateTime,
        default=lambda: datetime.now(timezone.utc),
        onupdate=lambda: datetime.now(timezone.utc),
    )

    interactions = relationship("Interaction", back_populates="person")
    network = relationship("MiraNetwork", back_populates="persons")
    conversations = relationship(
        "Conversation",
        back_populates="persons",
        secondary=person_conversation_association,
    )
    actions = relationship("Action", back_populates="person")
