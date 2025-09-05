"""
SQLAlchemy base configuration.
"""

from sqlalchemy.orm import DeclarativeBase


class Base(DeclarativeBase):
    """Base class for all SQLAlchemy models."""
    pass


# Import all models to register them with SQLAlchemy
from app.models.network import MiraNetwork  # noqa
from app.models.action import Action  # noqa
from app.models.interaction import Interaction  # noqa