"""
AI Assistant endpoints.
"""

from fastapi import APIRouter, Depends, HTTPException, UploadFile, File, status
from sqlalchemy.orm import Session
import uuid

from app.api.deps import get_db_dependency, get_network_auth_dependency
from app.models.network import MiraNetwork
from app.models.interaction import Interaction as InteractionModel
from app.services.command_service import CommandProcessor, WakeWordDetector
from app.services.inference_service import InferenceProcessor
from app.services.context_service import ContextProcessor

router = APIRouter()

# Initialize services
command_processor = CommandProcessor()
wake_word_detector = WakeWordDetector()
inference_processor = InferenceProcessor()
context_processor = ContextProcessor()


@router.post("/networks/{network_id}/interactions/register")
async def register_interaction(
    network_id: str,
    password: str,
    audio: UploadFile = File(...),
    network: MiraNetwork = Depends(get_network_auth_dependency()),
    db: Session = Depends(get_db_dependency)
):
    """Register interaction - transcribe sentence and process commands."""
    if not network.service_enabled:  # type: ignore
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Network service is currently disabled"
        )

    if len(await audio.read()) == 0:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Received empty audio data"
        )

    # Reset file pointer
    await audio.seek(0)
    sentence_buf_raw = await audio.read()

    try:
        # Mock interaction for now
        interaction_data = {
            "text": "Mock transcription",
            "network_id": network.id
        }

        interaction = InteractionModel(**interaction_data)
        db.add(interaction)
        db.commit()
        db.refresh(interaction)

        # Check for wake words
        wake_word_detection = wake_word_detector.detect_wake_words_text(
            client_id=str(network.id),
            transcribed_text=interaction_data["text"],
            audio_length=len(sentence_buf_raw) / (16000 * 2),
        )

        if wake_word_detection:
            if wake_word_detection.callback:
                db.delete(interaction)
                db.commit()
                return None

            response = command_processor.process_command(interaction)
            db.delete(interaction)
            db.commit()

            if response:
                return {"message": response}

        return interaction

    except Exception as e:
        db.rollback()
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(e)
        )


@router.get("/networks/{network_id}/interactions/{interaction_id}")
async def get_interaction(
    network_id: str,
    interaction_id: str,
    password: str,
    network: MiraNetwork = Depends(get_network_auth_dependency()),
    db: Session = Depends(get_db_dependency)
):
    """Get a specific interaction."""
    interaction = (
        db.query(InteractionModel)
        .filter(
            InteractionModel.id == uuid.UUID(interaction_id),
            InteractionModel.network_id == network.id
        )
        .first()
    )

    if not interaction:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Interaction not found"
        )

    return interaction


@router.get("/networks/{network_id}/interactions")
async def list_interactions(
    network_id: str,
    password: str,
    network: MiraNetwork = Depends(get_network_auth_dependency()),
    db: Session = Depends(get_db_dependency)
):
    """List recent interactions."""
    interactions = (
        db.query(InteractionModel)
        .filter(InteractionModel.network_id == network.id)
        .order_by(InteractionModel.created_at.desc())
        .limit(50)
        .all()
    )

    return interactions