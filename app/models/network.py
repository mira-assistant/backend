"""
MiraNetwork model for network management.
"""

from sqlalchemy import Column, String, DateTime, Boolean, JSON
from sqlalchemy.orm import relationship
from sqlalchemy.dialects.postgresql import UUID
from datetime import datetime, timezone
import uuid

from db.base import Base


class MiraNetwork(Base):
    """Represents a Mira network instance."""

    __tablename__ = "mira_networks"

    name = Column(String, nullable=False)
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)

    # Network status
    service_enabled = Column(Boolean, default=False)

    # Network settings
    network_settings = Column(JSON, default={})  # Changed from settings to network_settings

    # Relationships
    connected_clients = Column(JSON, default={})

    interactions = relationship(
        "Interaction", back_populates="network", cascade="all, delete-orphan"
    )

    # Timestamps
    created_at = Column(DateTime, default=lambda: datetime.now(timezone.utc))
    updated_at = Column(
        DateTime,
        default=lambda: datetime.now(timezone.utc),
        onupdate=lambda: datetime.now(timezone.utc),
    )




