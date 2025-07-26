from sqlalchemy import Column, String, DateTime, Integer
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.dialects.postgresql import UUID
import uuid
from datetime import datetime, timezone

Base = declarative_base()

class Interaction(Base):
    __tablename__ = "interactions"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    speaker = Column(Integer, nullable=False)
    text = Column(String, nullable=False)
    timestamp = Column(DateTime, default=lambda: datetime.now(timezone.utc))

# class User(Base):
#     __tablename__ = "users"

#     id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
#     name = Column(String, nullable=False)

class Conversation(Base):
    __tablename__ = "conversations"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    user_ids = Column(UUID(as_uuid=True), nullable=False)
    start_of_conversation = Column(DateTime, default=lambda: datetime.now(timezone.utc))
    end_of_conversation = Column(DateTime, nullable=True)


class Action(Base):
    __tablename__ = "actions"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    user_id = Column(UUID(as_uuid=True), nullable=False)
    action_type = Column(String, nullable=False)
    details = Column(String, nullable=True)