from sqlalchemy import Column, String, DateTime, Integer
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.dialects.postgresql import UUID
import uuid
from datetime import datetime, timezone
from sqlalchemy import ForeignKey
from sqlalchemy.orm import relationship

Base = declarative_base()

class Conversation(Base):
    """
    Represents a conversation entity in the database.
    Attributes:
        id (UUID): Unique identifier for the conversation.
        speakers (int): Number of speakers involved in the conversation.
        description (str, optional): Optional description of the conversation.
        interactions (List[Interaction]): List of interactions associated with this conversation.
    References:
        See also: Interaction model for related interactions.
    """

    
    __tablename__ = "conversations"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    speakers = Column(Integer, nullable=False)
    description = Column(String, nullable=True)
    interactions = relationship("Interaction", back_populates="conversation")


class Interaction(Base):
    """
    Represents an interaction entity in the database.
    Attributes:
        id (UUID): Unique identifier for the interaction.
        speaker (int): Identifier for the speaker involved in the interaction.
        text (str): Text content of the interaction.
        timestamp (datetime): Timestamp of when the interaction occurred.
        conversation_id (UUID): Foreign key referencing the associated conversation.
    """

    __tablename__ = "interactions"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    speaker = Column(Integer, nullable=False)
    text = Column(String, nullable=False)
    timestamp = Column(DateTime, default=lambda: datetime.now(timezone.utc))
    
    conversation_id = Column(UUID(as_uuid=True), ForeignKey("conversations.id"), nullable=False)
    conversation = relationship("Conversation", back_populates="interactions")


# class Action(Base):
#     """
#     Represents an action entity in the database.
#     Attributes:
#         id (UUID): Unique identifier for the action.
#         user_id (UUID): Identifier for the user who performed the action.
#         action_type (str): Type of the action performed.
#     """

#     __tablename__ = "actions"

#     id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
#     user_id = Column(UUID(as_uuid=True), nullable=False)
#     action_type = Column(String, nullable=False)
#     details = Column(String, nullable=True)