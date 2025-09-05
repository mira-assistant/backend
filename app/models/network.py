"""
MiraNetwork model for network management.
"""

from sqlalchemy import Column, String, DateTime, Boolean, JSON
from sqlalchemy.orm import relationship
from sqlalchemy.dialects.postgresql import UUID
from datetime import datetime, timezone
import uuid

from app.db.base import Base
from app.core.security import get_password_hash, verify_password


class MiraNetwork(Base):
    """Represents a Mira network instance."""

    __tablename__ = "mira_networks"

    name = Column(String, nullable=False)
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    hashed_password = Column(String, nullable=False)

    # Network status
    service_enabled = Column(Boolean, default=False)

    # Network settings
    network_settings = Column(JSON, default={})  # Changed from settings to network_settings

    # Relationships
    connected_clients = relationship(
        "NetworkClient", back_populates="network", cascade="all, delete-orphan"
    )
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

    @classmethod
    def create(cls, name: str, password: str) -> "MiraNetwork":
        """Create a new network."""
        return cls(network_id=uuid.uuid4(), name=name, hashed_password=get_password_hash(password))

    def verify_password(self, password: str) -> bool:
        """Verify network password."""
        try:
            stored_hash = str(self.hashed_password)
            return verify_password(password, stored_hash)
        except (TypeError, ValueError):
            return False

    def update_password(self, new_password: str) -> None:
        """Update network password."""
        self.hashed_password = get_password_hash(new_password)
