"""
MiraNetwork model for network management.
"""

import uuid
from datetime import datetime, timezone

from sqlalchemy import JSON, Boolean, Column, DateTime, String
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.orm import relationship

from db.base import Base


class MiraNetwork(Base):
    """Represents a Mira network instance."""

    __tablename__ = "mira_networks"

    name = Column(String, nullable=False)
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)

    # Network status
    service_enabled = Column(Boolean, default=False)

    # Network settings
    network_settings = Column(
        JSON, default={}
    )  # Changed from settings to network_settings

    # Relationships
    connected_clients = Column(JSON, default={})

    interactions = relationship(
        "Interaction", back_populates="network", cascade="all, delete-orphan"
    )

    persons = relationship("Person", back_populates="network")
    conversations = relationship("Conversation", back_populates="network")
    actions = relationship("Action", back_populates="network")

    # Timestamps
    created_at = Column(DateTime, default=lambda: datetime.now(timezone.utc))
    updated_at = Column(
        DateTime,
        default=lambda: datetime.now(timezone.utc),
        onupdate=lambda: datetime.now(timezone.utc),
    )
