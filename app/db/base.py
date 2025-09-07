"""
SQLAlchemy base configuration.
"""

from sqlalchemy.orm import DeclarativeBase


class Base(DeclarativeBase):
    """Base class for all SQLAlchemy models."""

    pass


# Import all models to register them with SQLAlchemy
from app.models import *