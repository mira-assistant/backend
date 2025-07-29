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
import uuid
from datetime import datetime, timezone
from sqlalchemy.orm import DeclarativeBase


class Base(DeclarativeBase):
    pass


class Person(Base):
    """Person entity for speaker recognition and management."""

    __tablename__ = "persons"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    name = Column(String, nullable=True)  # Can be None for unidentified speakers
    speaker_index = Column(
        Integer, unique=True, nullable=False
    )  # Original speaker number (1, 2, etc.)
    voice_embedding = Column(JSON, nullable=True)  # Store voice embedding as JSON array
    cluster_id = Column(Integer, nullable=True)  # DBSCAN cluster assignment
    created_at = Column(DateTime, default=lambda: datetime.now(timezone.utc))
    updated_at = Column(
        DateTime,
        default=lambda: datetime.now(timezone.utc),
        onupdate=lambda: datetime.now(timezone.utc),
    )

    # Relationships
    interactions = relationship("Interaction", back_populates="person")
    conversations = relationship("Conversation", back_populates="speaker")


class Interaction(Base):
    """Interaction entity for storing conversation interactions."""

    __tablename__ = "interactions"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    speaker_id = Column(
        UUID(as_uuid=True), ForeignKey("persons.id"), nullable=True
    )  # Link to Person
    text = Column(String, nullable=False)
    timestamp = Column(DateTime, default=lambda: datetime.now(timezone.utc))
    conversation_id = Column(UUID(as_uuid=True), ForeignKey("conversations.id"), nullable=True)

    # NLP-extracted features
    entities = Column(JSON, nullable=True)  # Named entities extracted from text
    topics = Column(JSON, nullable=True)  # Topic modeling results
    sentiment = Column(Float, nullable=True)  # Sentiment score

    # Relationships
    person = relationship("Person", back_populates="interactions")
    conversation = relationship("Conversation", back_populates="interactions")


class Conversation(Base):
    __tablename__ = "conversations"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    user_ids = Column(UUID(as_uuid=True), nullable=False)  # Keep for backward compatibility
    speaker_id = Column(
        UUID(as_uuid=True), ForeignKey("persons.id"), nullable=True
    )  # Primary speaker
    start_of_conversation = Column(DateTime, default=lambda: datetime.now(timezone.utc))
    end_of_conversation = Column(DateTime, nullable=True)

    # Enhanced conversation features
    topic_summary = Column(Text, nullable=True)  # AI-generated topic summary
    context_summary = Column(Text, nullable=True)  # Condensed long-term context
    participants = Column(JSON, nullable=True)  # List of person IDs in conversation

    # Relationships
    speaker = relationship("Person", back_populates="conversations")
    interactions = relationship("Interaction", back_populates="conversation")


class Action(Base):
    __tablename__ = "actions"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    user_id = Column(UUID(as_uuid=True), nullable=False)  # Keep for backward compatibility
    person_id = Column(
        UUID(as_uuid=True), ForeignKey("persons.id"), nullable=True
    )  # Link to Person
    action_type = Column(String, nullable=False)
    details = Column(String, nullable=True)
    interaction_id = Column(UUID(as_uuid=True), ForeignKey("interactions.id"), nullable=True)
    conversation_id = Column(UUID(as_uuid=True), ForeignKey("conversations.id"), nullable=True)

    # Enhanced action tracking
    status = Column(String, default="pending")  # pending, completed, failed
    scheduled_time = Column(DateTime, nullable=True)
    completed_time = Column(DateTime, nullable=True)
