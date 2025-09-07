from sqlalchemy import (
    Column,
    String,
    DateTime,
    Integer,
    JSON,
)
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.orm import relationship

import uuid
from datetime import datetime, timezone
from db.base import Base


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
