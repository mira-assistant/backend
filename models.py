from sqlalchemy import (
    Column,
    String,
    DateTime,
    Integer,
    Text,
    ForeignKey,
    JSON,
    Float,
)
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.orm import relationship
from sqlalchemy.orm import object_session

import typing

if typing.TYPE_CHECKING:
    from .models import Conversation
import uuid
from datetime import datetime, timezone
from sqlalchemy.orm import DeclarativeBase


class Base(DeclarativeBase):
    pass


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

    @property
    def conversations(self):
        session = object_session(self)
        if session is None:
            return []
        Conversation = self.__class__.__module__
        Conversation = globals().get("Conversation")
        if Conversation is None:
            from .models import Conversation
        return (
            session.query(Conversation)
            .filter(
                Conversation.user_ids.isnot(None), Conversation.user_ids.contains([str(self.id)])
            )
            .all()
        )


class Interaction(Base):
    """Interaction entity for storing conversation interactions."""

    __tablename__ = "interactions"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)

    text = Column(String, nullable=False)
    timestamp = Column(DateTime, default=lambda: datetime.now(timezone.utc))
    conversation_id = Column(UUID(as_uuid=True), ForeignKey("conversations.id"), nullable=True)

    voice_embedding = Column(JSON, nullable=True)
    speaker_id = Column(UUID(as_uuid=True), ForeignKey("persons.id"), nullable=True)

    text_embedding = Column(JSON, nullable=True)
    entities = Column(JSON, nullable=True)
    topics = Column(JSON, nullable=True)
    sentiment = Column(Float, nullable=True)

    person = relationship("Person", back_populates="interactions")
    conversation = relationship("Conversation", back_populates="interactions")


class Conversation(Base):
    __tablename__ = "conversations"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    user_ids = Column(JSON, nullable=True, default=lambda: list())

    topic_summary = Column(Text, nullable=True)
    context_summary = Column(Text, nullable=True)

    interactions = relationship("Interaction", back_populates="conversation")


class Action(Base):
    __tablename__ = "actions"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    user_id = Column(UUID(as_uuid=True), nullable=False)
    person_id = Column(UUID(as_uuid=True), ForeignKey("persons.id"), nullable=True)
    action_type = Column(String, nullable=False)
    details = Column(String, nullable=True)
    interaction_id = Column(UUID(as_uuid=True), ForeignKey("interactions.id"), nullable=True)
    conversation_id = Column(UUID(as_uuid=True), ForeignKey("conversations.id"), nullable=True)

    status = Column(String, default="pending")
    scheduled_time = Column(DateTime, nullable=True)
    completed_time = Column(DateTime, nullable=True)
