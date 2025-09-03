"""
AI Assistant endpoints.
"""
from fastapi import APIRouter, Depends, HTTPException, UploadFile, File, Form
from sqlalchemy.orm import Session
import uuid

from app.api.deps import get_db_dependency, get_optional_current_user
from app.schemas.interaction import Interaction, InteractionCreate
from app.schemas.action import Action
from app.models.interaction import Interaction as InteractionModel
from app.models.person import Person
from app.services.command_service import CommandProcessor, WakeWordDetector
from app.services.inference_service import InferenceProcessor
from app.services.context_service import ContextProcessor

router = APIRouter()

# Initialize services
command_processor = CommandProcessor()
wake_word_detector = WakeWordDetector()
inference_processor = InferenceProcessor()
context_processor = ContextProcessor()


@router.post("/interactions/register")
async def register_interaction(
    audio: UploadFile = File(...),
    client_id: str = Form(...),
    db: Session = Depends(get_db_dependency)
):
    """Register interaction - transcribe sentence, identify speaker, and update stream quality."""

    if len(await audio.read()) == 0:
        raise HTTPException(
            status_code=400,
            detail="Received empty audio data. Please provide valid audio."
        )

    # Reset file pointer
    await audio.seek(0)
    sentence_buf_raw = await audio.read()

    try:
        # This would need to be implemented with the actual audio processing
        # For now, we'll create a mock interaction
        interaction_data = {
            "text": "Mock transcription",  # This would come from actual transcription
            "speaker_id": None,  # This would come from speaker identification
        }

        interaction = InteractionModel(**interaction_data)
        db.add(interaction)
        db.commit()
        db.refresh(interaction)

        # Check for wake words
        speaker = db.query(Person).filter_by(id=interaction.speaker_id).first()

        if speaker and speaker.index == 1:  # Assuming index 1 is the primary user
            wake_word_detection = wake_word_detector.detect_wake_words_text(
                client_id=client_id,
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
        raise HTTPException(status_code=500, detail=f"Database error: {str(e)}")


@router.get("/interactions/{interaction_id}")
def get_interaction(
    interaction_id: str,
    db: Session = Depends(get_db_dependency)
):
    """Get a specific interaction by ID."""
    try:
        interaction = db.query(InteractionModel).filter_by(id=uuid.UUID(interaction_id)).first()

        if not interaction:
            raise HTTPException(status_code=404, detail="Interaction not found")

        return interaction
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to fetch interaction: {str(e)}")


@router.post("/interactions/{interaction_id}/inference")
def interaction_inference(
    interaction_id: str,
    db: Session = Depends(get_db_dependency)
):
    """Inference endpoint with database-backed context integration."""

    try:
        interaction = db.query(InteractionModel).filter_by(id=uuid.UUID(interaction_id)).first()

        if not interaction:
            raise HTTPException(status_code=404, detail="Interaction not found")

        context, has_intent = context_processor.build_context(interaction, db)

        if not has_intent:
            return {"message": "Intent not recognized, no inference performed."}

        action = inference_processor.extract_action(interaction=interaction, context=context)

        return action

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Inference failed: {str(e)}")


@router.delete("/interactions/{interaction_id}")
def delete_interaction(
    interaction_id: str,
    db: Session = Depends(get_db_dependency)
):
    """Delete a specific interaction from the database."""
    try:
        interaction = db.query(InteractionModel).filter_by(id=uuid.UUID(interaction_id)).first()
        if not interaction:
            raise HTTPException(status_code=404, detail="Interaction not found")

        db.delete(interaction)
        db.commit()

        return {"detail": "Interaction deleted successfully"}
    except Exception as e:
        db.rollback()
        raise HTTPException(status_code=500, detail=f"Failed to delete interaction: {str(e)}")

