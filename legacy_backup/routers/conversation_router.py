from mira import (
    logger,
)
from db import get_db_session
from models import Conversation
from fastapi import APIRouter, HTTPException

router = APIRouter(prefix="/conversations")


@router.get("/all")
def get_conversations():
    """Get recent conversations with their interactions."""
    try:
        db = get_db_session()
        try:
            conversations = (
                db.query(Conversation)
                .join(Interaction, Conversation.id == Interaction.conversation_id)
                .group_by(Conversation.id)
                .order_by(func.max(Interaction.timestamp).desc())
                .all()
            )

            return conversations

        finally:
            db.close()

    except Exception as e:
        logger.error(f"Error fetching conversations: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to fetch conversations: {str(e)}")
