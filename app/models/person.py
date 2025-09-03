"""
Person model for speaker recognition and management.
"""
from sqlalchemy import Column, String, DateTime, Integer, JSON
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.orm import relationship, object_session
from datetime import datetime, timezone
import uuid
from typing import TYPE_CHECKING

from app.db.base import Base

if TYPE_CHECKING:
    from app.models.conversation import Conversation


class Person(Base):
    """Person entity for speaker recognition and management."""

    __tablename__ = "persons"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    name = Column(String, nullable=True)
    index = Column(Integer, unique=True, nullable=False)

    voice_embedding = Column(JSON, nullable=True)
    cluster_id = Column(Integer, nullable=True)

    created_at = Column(DateTime, default=lambda: datetime.now(timezone.utc))
    updated_at = Column(
        DateTime,
        default=lambda: datetime.now(timezone.utc),
        onupdate=lambda: datetime.now(timezone.utc),
    )

    interactions = relationship("Interaction", back_populates="person")

    @property
    def conversations(self):
        """Get conversations for this person."""
        session = object_session(self)
        if session is None:
            return []

        from app.models.conversation import Conversation
        return (
            session.query(Conversation)
            .filter(
                Conversation.user_ids.isnot(None),
                Conversation.user_ids.contains([str(self.id)])
            )
            .all()
        )

