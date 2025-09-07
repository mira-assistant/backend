from sqlalchemy.orm import Session
from app.db import get_db
from app.models import Conversation
from fastapi import APIRouter, Depends, HTTPException, Path

router = APIRouter(prefix="/{network_id}/conversations")


@router.get("/{conversation_id}")
def get_conversation(
    conversation_id: str = Path(..., description="The ID of the conversation"),
    network_id: str = Path(..., description="The ID of the network"),
    db: Session = Depends(get_db),
):
    """Get a conversation by ID."""

    conversation = (
        db.query(Conversation)
        .filter(Conversation.network_id == network_id)
        .filter(Conversation.id == conversation_id)
        .first()
    )

    if not conversation:
        raise HTTPException(status_code=404, detail="Conversation not found")

    return conversation
