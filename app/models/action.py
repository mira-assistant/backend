"""
Action model for task management.
"""

from sqlalchemy import Column, String, DateTime
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.orm import relationship
import uuid

from app.db.base import Base


class Action(Base):
    __tablename__ = "actions"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)

    action_type = Column(String, nullable=False)
    details = Column(String, nullable=True)
    status = Column(String, default="created")

    interaction = relationship("Interaction", back_populates="actions")
    conversation = relationship("Conversation", back_populates="actions")
    network = relationship("MiraNetwork", back_populates="actions")

    scheduled_time = Column(DateTime, nullable=True)
    completed_time = Column(DateTime, nullable=True)
