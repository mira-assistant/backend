"""
Action model for storing user actions and tasks.
"""

from sqlalchemy import Column, String, DateTime, ForeignKey
from sqlalchemy.dialects.postgresql import UUID
import uuid
from typing import TYPE_CHECKING

from app.db.base import Base

if TYPE_CHECKING:
    from app.models.person import Person
    from app.models.interaction import Interaction
    from app.models.conversation import Conversation


class Action(Base):
    """Action entity for storing user actions and tasks."""

    __tablename__ = "actions"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    user_id = Column(UUID(as_uuid=True), nullable=False)
    person_id = Column(UUID(as_uuid=True), ForeignKey("persons.id"), nullable=True)
    action_type = Column(String, nullable=False)
    details = Column(String, nullable=True)
    interaction_id = Column(UUID(as_uuid=True), ForeignKey("interactions.id"), nullable=True)
    conversation_id = Column(UUID(as_uuid=True), ForeignKey("conversations.id"), nullable=True)

    status = Column(String, default="pending")
    scheduled_time = Column(DateTime, nullable=True)
    completed_time = Column(DateTime, nullable=True)
