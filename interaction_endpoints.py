import uuid
import logging
from fastapi import APIRouter, File, HTTPException, UploadFile, Form
from db import get_db_session_context, handle_db_error
from models import Interaction, Person
from processors import sentence_processor as SentenceProcessor

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/interactions", tags=["interactions"])


def get_status():
    from mira import status
    return status


def get_processors():
    from mira import context_processor, audio_scorer, wake_word_detector, command_processor, inference_processor
    return context_processor, audio_scorer, wake_word_detector, command_processor, inference_processor


@router.post("/register")
@handle_db_error("register_interaction")
async def register_interaction(audio: UploadFile = File(...), client_id: str = Form(...)):
    """Register interaction - transcribe sentence, identify speaker, and update stream quality."""

    status = get_status()
    context_processor, audio_scorer, wake_word_detector, command_processor, inference_processor = get_processors()

    if not status["enabled"]:
        raise HTTPException(status_code=503, detail="Service is currently disabled")

    if not client_id:
        raise HTTPException(
            status_code=400, detail="client_id is required for interaction registration"
        )

    sentence_buf_raw = await audio.read()

    with get_db_session_context() as db:
        if len(sentence_buf_raw) == 0:
            raise HTTPException(
                status_code=400, detail="Received empty audio data. Please provide valid audio."
            )

        logger.info(
            f"Processing audio data: {len(sentence_buf_raw)} bytes from client: {client_id}"
        )

        best_stream_info = audio_scorer.get_best_stream()
        audio_float = SentenceProcessor.pcm_bytes_to_float32(sentence_buf_raw)
        audio_scorer.update_stream_quality(client_id, audio_float)

        if not best_stream_info:
            raise HTTPException(
                status_code=404,
                detail="No active audio streams found. Please ensure clients are registered.",
            )

        if best_stream_info["client_id"] != client_id:
            logger.info(
                f"Interaction from {client_id} not registered - better stream available from {best_stream_info['client_id']} with score {best_stream_info['score']:.2f}"
            )

            return {
                "message": "Interaction was not registered due to better audio streams",
            }

        sentence_buf = bytearray(sentence_buf_raw)
        transcription_result = SentenceProcessor.transcribe_interaction(sentence_buf, True)
        print(f"Transcription result: {transcription_result}")

        interaction = Interaction(**transcription_result)

        db.add(interaction)
        db.flush()

        logger.info(f"Interaction saved to database with ID: {interaction.id}")
        status["recent_interactions"].append(interaction.id)

        speaker = None
        if interaction.speaker_id:
            speaker = db.query(Person).filter_by(id=interaction.speaker_id).first()

        if not speaker and interaction.speaker_id:
            raise HTTPException(status_code=404, detail="Speaker not found")

        if speaker and speaker.index == 1:
            wake_word_detection = wake_word_detector.detect_wake_words_text(
                client_id=client_id,
                transcribed_text=transcription_result["text"],
                audio_length=len(sentence_buf_raw) / (16000 * 2),
            )

            if wake_word_detection:
                logger.info(
                    f"Wake word '{wake_word_detection.wake_word}' detected in audio from client {client_id}"
                )

                if wake_word_detection.callback:
                    return None

                response = command_processor.process_command(interaction=interaction)

                db.delete(interaction)
                db.flush()

                if response:
                    return {"message": response}

        return interaction


@router.get("/{interaction_id}")
@handle_db_error("get_interaction")
def get_interaction(interaction_id: str):
    """Get a specific interaction by ID."""
    with get_db_session_context() as db:
        interaction = db.query(Interaction).filter_by(id=uuid.UUID(interaction_id)).first()

        if not interaction:
            raise HTTPException(status_code=404, detail="Interaction not found")

        return interaction


@router.post("/{interaction_id}/inference")
@handle_db_error("interaction_inference")
def interaction_inference(interaction_id: str):
    """Inference endpoint with database-backed context integration."""

    status = get_status()
    context_processor, audio_scorer, wake_word_detector, command_processor, inference_processor = get_processors()

    if not status["enabled"]:
        raise HTTPException(status_code=503, detail="Service is currently disabled")

    with get_db_session_context() as db:
        interaction = db.query(Interaction).filter_by(id=uuid.UUID(interaction_id)).first()

        if not interaction:
            raise HTTPException(status_code=404, detail="Interaction not found")

        context, has_intent = context_processor.build_context(interaction)

        if not has_intent:
            return {"message": "Intent not recognized, no inference performed."}

        action = inference_processor.extract_action(interaction=interaction, context=context)

        return action


@router.get("", deprecated=True)
@handle_db_error("get_interactions")
def get_interactions(limit: int = 0):
    """Get recent interactions for live interaction display."""
    with get_db_session_context() as db:
        query = db.query(Interaction).order_by(Interaction.timestamp.desc())
        if limit != 0:
            query = query.limit(limit)
        interactions = query.all()

        return interactions


@router.delete("/{interaction_id}")
@handle_db_error("delete_interaction")
def delete_interaction(interaction_id: str):
    """Delete a specific interaction from the database."""
    status = get_status()

    with get_db_session_context() as db:
        interaction = db.query(Interaction).filter_by(id=uuid.UUID(interaction_id)).first()
        if not interaction:
            raise HTTPException(status_code=404, detail="Interaction not found")

        db.delete(interaction)
        db.flush()

        status["recent_interactions"].remove(interaction.id)

        return {"detail": "Interaction deleted successfully"}