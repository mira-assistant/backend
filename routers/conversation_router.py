from mira import status, context_processor, audio_scorer, wake_word_detector, command_processor, inference_processor, logger
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
                .order_by(Conversation.interactions[0].timestamp.desc())
                .all()
            )

            return conversations

        finally:
            db.close()

    except Exception as e:
        logger.error(f"Error fetching conversations: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to fetch conversations: {str(e)}")