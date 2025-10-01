import uuid
from datetime import datetime, timezone

from sqlalchemy import Column, DateTime, ForeignKey, String
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.orm import relationship

from app.db.base import Base


class Action(Base):
    __tablename__ = "actions"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    user_id = Column(UUID(as_uuid=True), nullable=False)
    person_id = Column(UUID(as_uuid=True), ForeignKey("persons.id"), nullable=True)
    action_type = Column(String, nullable=False)
    details = Column(String, nullable=True)
    interaction_id = Column(
        UUID(as_uuid=True), ForeignKey("interactions.id"), nullable=True
    )
    conversation_id = Column(
        UUID(as_uuid=True), ForeignKey("conversations.id"), nullable=True
    )

    # Foreign key to network
    network_id = Column(
        UUID(as_uuid=True), ForeignKey("mira_networks.id"), nullable=False
    )

    status = Column(String, default="pending")
    scheduled_time = Column(DateTime, nullable=True)
    completed_time = Column(DateTime, nullable=True)

    # Timestamps
    created_at = Column(DateTime, default=lambda: datetime.now(timezone.utc))
    updated_at = Column(
        DateTime,
        default=lambda: datetime.now(timezone.utc),
        onupdate=lambda: datetime.now(timezone.utc),
    )

    # Relationships
    person = relationship("Person", back_populates="actions")
    interaction = relationship("Interaction", back_populates="actions")
    conversation = relationship("Conversation", back_populates="actions")
    network = relationship("MiraNetwork", back_populates="actions")
