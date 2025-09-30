import uuid

from fastapi import APIRouter, Depends, HTTPException, Path
from sqlalchemy.orm import Session

from db import get_db
from models import Conversation

router = APIRouter(prefix="/conversations")


@router.get("/{conversation_id}")
def get_conversation(
    conversation_id: str = Path(..., description="The ID of the conversation"),
    network_id: str = Path(..., description="The ID of the network"),
    db: Session = Depends(get_db),
):
    """Get a conversation by ID."""

    try:
        network_uuid = uuid.UUID(network_id)
        conversation_uuid = uuid.UUID(conversation_id)
    except ValueError:
        raise HTTPException(status_code=422, detail="Invalid UUID format")

    conversation = (
        db.query(Conversation)
        .filter(Conversation.network_id == network_uuid)
        .filter(Conversation.id == conversation_uuid)
        .first()
    )

    if not conversation:
        raise HTTPException(status_code=404, detail="Conversation not found")

    return conversation
