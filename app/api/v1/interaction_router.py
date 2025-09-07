from sqlalchemy.orm import Session

import db
import models
from fastapi import APIRouter, Depends, Path, UploadFile, File, Form, HTTPException

from core.mira_logger import MiraLogger
from services.sentence_processor import SentenceProcessor
from services.command_processor import CommandProcessor, WakeWordDetector
from services.context_processor import ContextProcessor
from services.inference_processor import InferenceProcessor

router = APIRouter(prefix="/{network_id}/interactions")


@router.post("/register")
async def register_interaction(
    network_id: str = Path(..., description="The ID of the network"),
    audio: UploadFile = File(..., description="The audio file to register"),
    client_id: str = Form(..., description="The ID of the client that recorded the interaction"),
    db: Session = Depends(db.get_db),
):
    """Register interaction - transcribe sentence, identify speaker, and update stream quality."""

    network = db.query(models.MiraNetwork).filter(models.MiraNetwork.id == network_id).first()
    if not network:
        raise HTTPException(status_code=404, detail="Network not found")

    if not client_id or client_id not in network.connected_clients:
        raise HTTPException(
            status_code=400,
            detail="client_id is required for interaction registration and must be registered with the network",
        )

    sentence_buf_raw = await audio.read()

    if len(sentence_buf_raw) == 0:
        raise HTTPException(
            status_code=400, detail="Received empty audio data. Please provide valid audio."
        )

    MiraLogger.info(
        f"Processing audio data: {len(sentence_buf_raw)} bytes from client: {client_id}"
    )

    # best_stream_info = audio_scorer.get_best_stream()
    # audio_float = sentence_processor.pcm_bytes_to_float32(sentence_buf_raw)
    # audio_scorer.update_stream_quality(client_id, audio_float)

    # if not best_stream_info:
    #     raise HTTPException(
    #         status_code=404,
    #         detail="No active audio streams found. Please ensure clients are registered.",
    #     )

    # if best_stream_info["client_id"] != client_id:
    #     MiraLogger.info(
    #         f"Interaction from {client_id} not registered - better stream available from {best_stream_info['client_id']} with score {best_stream_info['score']:.2f}"
    #     )

    #     return {
    #         "message": "Interaction was not registered due to better audio streams",
    #     }

    sentence_buf = bytearray(sentence_buf_raw)
    transcription_result = SentenceProcessor.transcribe_interaction(network_id, sentence_buf, True)

    interaction = models.Interaction(
        **transcription_result,
        network_id=network_id,
        client_id=client_id,
    )

    db.add(interaction)
    db.commit()
    db.refresh(interaction)

    MiraLogger.info(f"Interaction saved to database with ID: {interaction.id}")

    speaker = (
        db.query(models.Person)
        .filter(models.Person.network_id == network_id)
        .filter(models.Person.id == interaction.speaker_id)
        .first()
    )

    if not speaker:
        raise HTTPException(status_code=404, detail="Speaker not found")

    if speaker.index == 1:  # type: ignore
        wake_word_detection = WakeWordDetector.detect_wake_words_text(
            network_id=network_id,
            client_id=client_id,
            transcribed_text=transcription_result["text"],
            audio_length=len(sentence_buf_raw) / (16000 * 2),
        )

        if wake_word_detection:
            MiraLogger.info(
                f"Wake word '{wake_word_detection.wake_word}' detected in audio from client {client_id}"
            )

            if wake_word_detection.callback:
                db.delete(interaction)
                db.commit()
                return None

            response = CommandProcessor.process_command(
                interaction=interaction,
                network_id=network_id,
            )

            db.delete(interaction)
            db.commit()

            if response:
                return {"message": response}

        return interaction


@router.get("/{interaction_id}")
def get_interaction(
    interaction_id: str = Path(..., description="The ID of the interaction"),
    network_id: str = Path(..., description="The ID of the network"),
    db: Session = Depends(db.get_db),
):
    """Get a specific interaction by ID."""
    interaction = (
        db.query(models.Interaction)
        .filter(models.Interaction.network_id == network_id)
        .filter(models.Interaction.id == interaction_id)
        .first()
    )

    if not interaction:
        raise HTTPException(status_code=404, detail="Interaction not found")

    return interaction


@router.delete("/{interaction_id}")
def delete_interaction(
    interaction_id: str = Path(..., description="The ID of the interaction"),
    network_id: str = Path(..., description="The ID of the network"),
    db: Session = Depends(db.get_db),
):
    """Delete a specific interaction from the database."""
    interaction = (
        db.query(models.Interaction)
        .filter(models.Interaction.network_id == network_id)
        .filter(models.Interaction.id == interaction_id)
        .first()
    )

    if not interaction:
        raise HTTPException(status_code=404, detail="Interaction not found")

    db.delete(interaction)
    db.commit()

    return {"detail": "Interaction deleted successfully"}


@router.post("/{interaction_id}/inference")
def interaction_inference(
    interaction_id: str = Path(..., description="The ID of the interaction"),
    network_id: str = Path(..., description="The ID of the network"),
    db: Session = Depends(db.get_db),
):
    """Inference endpoint with database-backed context integration."""

    network = db.query(models.MiraNetwork).filter(models.MiraNetwork.id == network_id).first()
    if not network:
        raise HTTPException(status_code=404, detail="Network not found")

    interaction = (
        db.query(models.Interaction)
        .filter(models.Interaction.network_id == network_id)
        .filter(models.Interaction.id == interaction_id)
        .first()
    )

    if not interaction:
        raise HTTPException(status_code=404, detail="Interaction not found")

    context, has_intent = ContextProcessor.build_context(interaction, network_id)

    if not has_intent:
        return {"message": "Intent not recognized, no inference performed."}

    action = InferenceProcessor.extract_action(interaction=interaction, network_id=network_id, context=context)  # type: ignore

    return action
