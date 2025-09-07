from sqlalchemy.orm import Session
from db import get_db
from models import Conversation
from fastapi import APIRouter, Depends

router = APIRouter(prefix="/conversations")

@router.get("/{conversation_id}")
def get_conversation(conversation_id: str, db: Session = Depends(get_db)):
    """Get a conversation by ID."""

    conversation = db.query(Conversation).filter(Conversation.id == conversation_id).first()

    return conversation