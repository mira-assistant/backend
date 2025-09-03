"""
Conversation model for managing conversation sessions.
"""

from sqlalchemy import Column, Text, JSON
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.orm import relationship
import uuid
from typing import TYPE_CHECKING

from app.db.base import Base

if TYPE_CHECKING:
    from app.models.interaction import Interaction


class Conversation(Base):
    """Conversation entity for managing conversation sessions."""

    __tablename__ = "conversations"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    user_ids = Column(JSON, nullable=True, default=lambda: list())

    topic_summary = Column(Text, nullable=True)
    context_summary = Column(Text, nullable=True)

    interactions = relationship("Interaction", back_populates="conversation")
