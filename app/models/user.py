"""
User model for authentication system.
"""

import uuid
from datetime import datetime, timezone

from sqlalchemy import Boolean, Column, DateTime, String
from sqlalchemy.dialects.postgresql import UUID

from app.db.base import Base


class User(Base):
    """User entity for authentication and authorization."""

    __tablename__ = "users"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    username = Column(String, unique=True, nullable=True)
    email = Column(String, unique=True, nullable=False)
    hashed_password = Column(String, nullable=True)  # nullable for OAuth users

    # OAuth provider info
    google_id = Column(String, unique=True, nullable=True)
    github_id = Column(String, unique=True, nullable=True)

    # User status
    is_active = Column(Boolean, default=True)

    # Timestamps
    created_at = Column(DateTime, default=lambda: datetime.now(timezone.utc))
    updated_at = Column(
        DateTime,
        default=lambda: datetime.now(timezone.utc),
        onupdate=lambda: datetime.now(timezone.utc),
    )

    def __repr__(self):
        return f"<User(id={self.id}, email={self.email})>"
