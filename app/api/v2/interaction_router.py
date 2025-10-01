import uuid
from typing import Optional, Union

from fastapi import APIRouter, Depends, File, Form, HTTPException, Path, UploadFile
from sqlalchemy.orm import Session

import app.db as db
import app.models as models
from app.core.constants import SAMPLE_RATE
from app.core.mira_logger import MiraLogger
from app.services.service_factory import (
    get_command_processor,
    get_context_processor,
    get_inference_processor,
    get_multi_stream_processor,
    get_sentence_processor,
)

router = APIRouter(prefix="/interactions")


def _validate_network_and_client(
    network_id: str, client_id: str, db: Session
) -> tuple[uuid.UUID, models.MiraNetwork]:
    """Validate network ID and client ID, return network UUID and network object."""
    try:
        network_uuid = uuid.UUID(network_id)
    except ValueError:
        raise HTTPException(status_code=422, detail="Invalid UUID format")

    network = (
        db.query(models.MiraNetwork)
        .filter(models.MiraNetwork.id == network_uuid)
        .first()
    )
    if not network:
        raise HTTPException(status_code=404, detail="Network not found")

    if not client_id or client_id not in network.connected_clients:
        raise HTTPException(
            status_code=400,
            detail="client_id is required for interaction registration and must be registered with the network",
        )

    return network_uuid, network


async def _validate_audio_data(audio: UploadFile) -> bytes:
    """Validate and read audio data."""
    sentence_buf_raw = await audio.read()

    if len(sentence_buf_raw) == 0:
        raise HTTPException(
            status_code=400,
            detail="Received empty audio data. Please provide valid audio.",
        )

    return sentence_buf_raw


def _process_stream_quality(
    network_id: str, client_id: str, sentence_buf_raw: bytes
) -> None:
    """Process stream quality analysis and logging."""
    sentence_processor = get_sentence_processor(network_id)
    multi_stream_processor = get_multi_stream_processor(network_id)

    # Convert audio to float32 for stream quality analysis
    audio_float = sentence_processor.pcm_bytes_to_float32(sentence_buf_raw)

    # Update stream quality for this client
    stream_metrics = multi_stream_processor.update_stream_quality(
        client_id, audio_float
    )

    if stream_metrics:
        MiraLogger.info(
            f"Stream quality updated for {client_id}: SNR={stream_metrics.snr:.1f}dB, "
            f"Clarity={stream_metrics.speech_clarity:.1f}, Score={stream_metrics.score:.1f}"
        )


def _check_stream_quality(network_id: str, client_id: str) -> Optional[dict]:
    """Check if current client has the best stream quality."""
    multi_stream_processor = get_multi_stream_processor(network_id)
    best_stream_info = multi_stream_processor.get_best_stream()

    if not best_stream_info["client_id"]:
        MiraLogger.warning(f"No active audio streams found for network {network_id}")
        return None

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

    return None


def _process_transcription_and_save(
    network_id: str, client_id: str, sentence_buf_raw: bytes, db: Session
) -> models.Interaction:
    """Process audio transcription and save interaction to database."""
    sentence_processor = get_sentence_processor(network_id)
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
    return interaction


def _handle_wake_word_processing(
    interaction: models.Interaction,
    network_id: str,
    client_id: str,
    sentence_buf_raw: bytes,
    db: Session,
) -> Union[dict, models.Interaction, None]:
    """Handle wake word detection and command processing."""
    speaker = (
        db.query(models.Person)
        .filter(models.Person.network_id == interaction.network_id)
        .filter(models.Person.id == interaction.speaker_id)
        .first()
    )

    if not speaker:
        raise HTTPException(status_code=404, detail="Speaker not found")

    if speaker.index != 1:  # type: ignore
        return interaction

    command_processor = get_command_processor(network_id)
    wake_word_detection = command_processor.detect_wake_words_text(
        client_id=client_id,
        transcribed_text=interaction.text,
        audio_length=len(sentence_buf_raw) / (SAMPLE_RATE * 2),
    )

    if not wake_word_detection:
        return interaction

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


@router.post("/register")
async def register_interaction(
    network_id: str = Path(..., description="The ID of the network"),
    audio: UploadFile = File(..., description="The audio file to register"),
    client_id: str = Form(
        ..., description="The ID of the client that recorded the interaction"
    ),
    db: Session = Depends(db.get_db),
):
    """Register interaction - transcribe sentence, identify speaker, and update stream quality."""

    # Validate network and client
    network_uuid, network = _validate_network_and_client(network_id, client_id, db)

    # Validate and read audio data
    sentence_buf_raw = await _validate_audio_data(audio)

    MiraLogger.info(
        f"Processing audio data: {len(sentence_buf_raw)} bytes from client: {client_id}"
    )

    # Process stream quality
    _process_stream_quality(network_id, client_id, sentence_buf_raw)

    # Check if this client has the best stream quality
    stream_quality_response = _check_stream_quality(network_id, client_id)
    if stream_quality_response:
        return stream_quality_response

    # Process transcription and save interaction
    interaction = _process_transcription_and_save(
        network_id, client_id, sentence_buf_raw, db
    )

    # Handle wake word processing
    return _handle_wake_word_processing(
        interaction, network_id, client_id, sentence_buf_raw, db
    )


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

    network = (
        db.query(models.MiraNetwork)
        .filter(models.MiraNetwork.id == network_uuid)
        .first()
    )
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
    action = inference_processor.extract_action(
        interaction=interaction, context=context
    )  # type: ignore

    return action
