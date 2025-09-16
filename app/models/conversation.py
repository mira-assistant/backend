import uuid

from sqlalchemy import JSON, Column, ForeignKey, Table, Text
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.orm import relationship

from db.base import Base

# Association table for many-to-many relationship between Person and Conversation
person_conversation_association = Table(
    "person_conversation",
    Base.metadata,
    Column("person_id", UUID(as_uuid=True), ForeignKey("persons.id"), primary_key=True),
    Column(
        "conversation_id",
        UUID(as_uuid=True),
        ForeignKey("conversations.id"),
        primary_key=True,
    ),
)


class Conversation(Base):
    __tablename__ = "conversations"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    user_ids = Column(JSON, nullable=True, default=lambda: list())

    topic_summary = Column(Text, nullable=True)
    context_summary = Column(Text, nullable=True)

    # Foreign key to network
    network_id = Column(
        UUID(as_uuid=True), ForeignKey("mira_networks.id"), nullable=False
    )

    interactions = relationship("Interaction", back_populates="conversation")
    network = relationship("MiraNetwork", back_populates="conversations")
    persons = relationship(
        "Person",
        back_populates="conversations",
        secondary=person_conversation_association,
    )
    actions = relationship("Action", back_populates="conversation")
