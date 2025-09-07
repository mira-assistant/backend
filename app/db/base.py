"""
SQLAlchemy base configuration.
"""

from sqlalchemy.orm import DeclarativeBase


class Base(DeclarativeBase):
    """Base class for all SQLAlchemy models."""

    pass


# Models will be imported when needed to avoid circular imports