"""
NetworkClient model for secure client management.
"""

from sqlalchemy import Column, String, DateTime
from sqlalchemy.orm import relationship
from datetime import datetime, timezone

from app.db.base import Base


class MiraClient(Base):
    """Represents a client device connected to a Mira network."""

    __tablename__ = "network_clients"

    # Client identification
    identifier = Column(String, nullable=False)

    # Relationships
    network = relationship("MiraNetwork", back_populates="connected_clients")

    connection_start = Column(DateTime, nullable=False, default=datetime.now(timezone.utc))
