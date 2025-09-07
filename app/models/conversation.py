from sqlalchemy import (
    Column,
    Text,
    JSON,
)
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.orm import relationship

import uuid
from db.base import Base


class Conversation(Base):
    __tablename__ = "conversations"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    user_ids = Column(JSON, nullable=True, default=lambda: list())

    topic_summary = Column(Text, nullable=True)
    context_summary = Column(Text, nullable=True)

    interactions = relationship("Interaction", back_populates="conversation")
