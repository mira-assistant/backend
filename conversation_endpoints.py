import logging
from fastapi import APIRouter, HTTPException
from db import get_db_session_context, handle_db_error
from models import Conversation

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/conversations", tags=["conversations"])


@router.get("")
@handle_db_error("get_recent_conversations")
def get_recent_conversations(limit: int = 10):
    """Get recent conversations with their interactions."""
    with get_db_session_context() as db:
        conversations = (
            db.query(Conversation)
            .order_by(Conversation.interactions[0].timestamp.desc())
            .limit(limit)
            .all()
        )

        result = []
        for conv in conversations:
            conv_data = {
                "id": str(conv.id),
                "user_ids": [str(user_id) for user_id in conv.user_ids],
                "interaction_count": len(conv.interactions),
                "topic_summary": conv.topic_summary,
            }
            result.append(conv_data)

        return result