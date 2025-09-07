from sqlalchemy.orm import Session
import uuid

import app.db as db
import app.models as models
from fastapi import APIRouter, Depends, Path, UploadFile, File, Form, HTTPException

from app.core.mira_logger import MiraLogger
from app.services.service_factory import (
    get_sentence_processor,
    get_command_processor,
    get_context_processor,
    get_inference_processor,
    get_multi_stream_processor,
)

router = APIRouter(prefix="/{network_id}/interactions")


@router.post("/register")
async def register_interaction(
    network_id: str = Path(..., description="The ID of the network"),
    audio: UploadFile = File(..., description="The audio file to register"),
    client_id: str = Form(..., description="The ID of the client that recorded the interaction"),
    db: Session = Depends(db.get_db),
):
    """Register interaction - transcribe sentence, identify speaker, and update stream quality."""

    try:
        network_uuid = uuid.UUID(network_id)
    except ValueError:
        raise HTTPException(status_code=422, detail="Invalid UUID format")

    network = db.query(models.MiraNetwork).filter(models.MiraNetwork.id == network_uuid).first()
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

    # Get services for this network
    sentence_processor = get_sentence_processor(network_id)
    multi_stream_processor = get_multi_stream_processor(network_id)

    # Convert audio to float32 for stream quality analysis
    audio_float = sentence_processor.pcm_bytes_to_float32(sentence_buf_raw)

    # Update stream quality for this client
    stream_metrics = multi_stream_processor.update_stream_quality(client_id, audio_float)

    if stream_metrics:
        MiraLogger.info(
            f"Stream quality updated for {client_id}: SNR={stream_metrics.snr:.1f}dB, "
            f"Clarity={stream_metrics.speech_clarity:.1f}, Score={stream_metrics.score:.1f}"
        )

    # Get the best available stream
    best_stream_info = multi_stream_processor.get_best_stream()

    # Check if there are any active streams
    if not best_stream_info["client_id"]:
        MiraLogger.warning(f"No active audio streams found for network {network_id}")
    else:
        # Check if this client has the best stream quality
        if best_stream_info["client_id"] != client_id:
            MiraLogger.info(
                f"Interaction from {client_id} not registered - better stream available from "
                f"{best_stream_info['client_id']} with score {best_stream_info['score']:.2f}"
            )

            return {
                "message": "Interaction was not registered due to better audio streams",
                "best_client_id": best_stream_info["client_id"],
                "best_score": best_stream_info["score"],
                "current_client_id": client_id,
            }

    # Process the audio for transcription and speaker identification
    sentence_buf = bytearray(sentence_buf_raw)
    transcription_result = sentence_processor.transcribe_interaction(sentence_buf, True)

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
        .filter(models.Person.network_id == network_uuid)
        .filter(models.Person.id == interaction.speaker_id)
        .first()
    )

    if not speaker:
        raise HTTPException(status_code=404, detail="Speaker not found")

    if speaker.index == 1:  # type: ignore
        command_processor = get_command_processor(network_id)
        wake_word_detection = command_processor.detect_wake_words_text(
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

            response = command_processor.process_command(interaction)

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

    try:
        network_uuid = uuid.UUID(network_id)
        interaction_uuid = uuid.UUID(interaction_id)
    except ValueError:
        raise HTTPException(status_code=422, detail="Invalid UUID format")

    interaction = (
        db.query(models.Interaction)
        .filter(models.Interaction.network_id == network_uuid)
        .filter(models.Interaction.id == interaction_uuid)
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

    try:
        network_uuid = uuid.UUID(network_id)
        interaction_uuid = uuid.UUID(interaction_id)
    except ValueError:
        raise HTTPException(status_code=422, detail="Invalid UUID format")

    interaction = (
        db.query(models.Interaction)
        .filter(models.Interaction.network_id == network_uuid)
        .filter(models.Interaction.id == interaction_uuid)
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

    try:
        network_uuid = uuid.UUID(network_id)
        interaction_uuid = uuid.UUID(interaction_id)
    except ValueError:
        raise HTTPException(status_code=422, detail="Invalid UUID format")

    network = db.query(models.MiraNetwork).filter(models.MiraNetwork.id == network_uuid).first()
    if not network:
        raise HTTPException(status_code=404, detail="Network not found")

    interaction = (
        db.query(models.Interaction)
        .filter(models.Interaction.network_id == network_uuid)
        .filter(models.Interaction.id == interaction_uuid)
        .first()
    )

    if not interaction:
        raise HTTPException(status_code=404, detail="Interaction not found")

    context_processor = get_context_processor(network_id)
    context, has_intent = context_processor.build_context(interaction)

    if not has_intent:
        return {"message": "Intent not recognized, no inference performed."}

    inference_processor = get_inference_processor(network_id)
    action = inference_processor.extract_action(interaction=interaction, context=context)  # type: ignore

    return action
