import uuid
from fastapi import Body, FastAPI, File, HTTPException, UploadFile, Form, Request
from fastapi.middleware.cors import CORSMiddleware
import logging
import json
from collections import deque
from contextlib import asynccontextmanager
from datetime import datetime, timezone


from db import get_db_session
from models import Interaction, Person, Conversation


import sentence_processor as SentenceProcessor
from inference_processor import InferenceProcessor
from context_processor import ContextProcessor
from multi_stream_processor import MultiStreamProcessor
from command_processor import CommandProcessor, WakeWordDetector


# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

print("\n\n\n\n\n\nInitializing Mira backend service...")
context_processor = ContextProcessor()
audio_scorer = MultiStreamProcessor()
wake_word_detector = WakeWordDetector()
command_processor = CommandProcessor()
inference_processor = InferenceProcessor()

status: dict = {
    "version": "4.3.0",
    "connected_clients": dict(),
    "best_client": None,
    "enabled": False,
    "recent_interactions": deque(maxlen=10),
    "last_command_result": None,  # Store last command processing result
}

hosting_urls = {
    "localhost": "http://localhost:8000",
    "ankurs-macbook-air": "http://100.75.140.79:8000",
}


@asynccontextmanager
async def lifespan(app: FastAPI):
    for interaction in (
        get_db_session().query(Interaction).order_by(Interaction.timestamp.desc()).limit(10).all()
    ):
        status["recent_interactions"].append(interaction.id)

    # Initialize wake words
    wake_word_detector.add_wake_word("mira cancel", sensitivity=0.7, callback=disable_service)
    wake_word_detector.add_wake_word("mira exit", sensitivity=0.7, callback=disable_service)
    wake_word_detector.add_wake_word("mira quit", sensitivity=0.5, callback=disable_service)
    wake_word_detector.add_wake_word("mira stop", sensitivity=0.7, callback=disable_service)
    yield


# Initialize FastAPI app first
app = FastAPI(lifespan=lifespan)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins for development
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


def log_once(message, flag_name=None):
    """Log a message only once during initialization"""
    if flag_name == "advanced" and not globals().get("_advanced_logged", False):
        print(message)
        globals()["_advanced_logged"] = True
    elif flag_name == "context" and not globals().get("_context_logged", False):
        print(message)
        globals()["_context_logged"] = True
    elif flag_name is None and not globals().get("_general_logged", False):
        print(message)
        globals()["_general_logged"] = True


@app.get("/")
def root():
    scores = audio_scorer.get_all_stream_scores()
    status["best_client"] = audio_scorer.get_best_stream()

    # Update connection runtime for all connected clients
    current_time = datetime.now(timezone.utc)
    for client_id, client_info in status["connected_clients"].items():
        if "connection_start_time" in client_info:
            connection_start = client_info["connection_start_time"]
            if isinstance(connection_start, str):
                # Handle backward compatibility if stored as string
                connection_start = datetime.fromisoformat(connection_start.replace("Z", "+00:00"))

            runtime_seconds = (current_time - connection_start).total_seconds()
            client_info["connection_runtime"] = round(runtime_seconds, 2)

        # Add score information
        if client_id in scores:
            client_info["score"] = scores[client_id]

    return status


@app.post("/service/client/register/{client_id}")
def register_client(client_id: str, request: Request):
    """Register a client and initialize stream scoring."""
    # Get client IP address and connection start time
    client_ip = request.client.host if request.client else "unknown"
    connection_start_time = datetime.now(timezone.utc)

    # Store client information in connected_clients dictionary
    status["connected_clients"][client_id] = {
        "ip": client_ip,
        "connection_start_time": connection_start_time,
        "connection_runtime": 0.0,  # Runtime in seconds, updated dynamically
    }

    # Register client with audio stream scorer
    success = audio_scorer.register_client(client_id=client_id)

    if success:
        logger.info(f"Client {client_id} registered for stream scoring from IP {client_ip}")

    return {"message": f"{client_id} registered successfully", "stream_scoring_enabled": success}


@app.delete("/service/client/deregister/{client_id}")
def deregister_client(client_id: str):
    """Deregister a client and remove from stream scoring."""
    if client_id in status["connected_clients"]:
        del status["connected_clients"][client_id]
    else:
        print("Client already deregistered or not found:", client_id)

    # Deregister from audio stream scorer
    success = audio_scorer.deregister_client(client_id)

    if client_id not in status["connected_clients"] and not success:
        return {"message": f"{client_id} already deregistered or not found"}

    return {"message": f"{client_id} deregistered successfully", "stream_scoring_removed": success}


@app.patch("/service/enable")
def enable_service():
    status["enabled"] = True
    return {"message": "Service enabled successfully"}


@app.patch("/service/disable")
def disable_service():
    status["enabled"] = False
    return {"message": "Service disabled successfully"}


@app.post("/interactions/register")
async def register_interaction(audio: UploadFile = File(...), client_id: str = Form(...)):
    """Register interaction - transcribe sentence, identify speaker, and update stream quality."""

    if not client_id:
        raise HTTPException(
            status_code=400, detail="client_id is required for interaction registration"
        )

    sentence_buf_raw = await audio.read()

    db = get_db_session()

    try:
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

        # Check if this client has the best stream quality
        if best_stream_info["client_id"] != client_id:
            logger.info(
                f"Interaction from {client_id} not registered - better stream available from {best_stream_info['client_id']} with score {best_stream_info['score']:.2f}"
            )

            return {
                "message": "Interaction was not registered due to better audio streams",
            }

        sentence_buf = bytearray(sentence_buf_raw)
        transcription_result = SentenceProcessor.transcribe_interaction(sentence_buf)

        if not transcription_result or not transcription_result["text"]:
            raise HTTPException(
                status_code=400, detail="Transcription failed. Please check the audio quality."
            )

        try:
            interaction = Interaction(
                **transcription_result,
            )

            db.add(interaction)
            db.commit()
            db.refresh(interaction)

            logger.info(f"Interaction saved to database with ID: {interaction.id}")
            status["recent_interactions"].append(interaction.id)

            speaker = db.query(Person).filter_by(id=interaction.speaker_id).first()

            if not speaker:
                raise HTTPException(status_code=404, detail="Speaker not found")

            if speaker.index == 1:  # type: ignore
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

                    response = command_processor.process_command(
                        interaction=interaction,
                    )

                    if response:
                        return {"message": response}

            return interaction

        except Exception as db_error:
            logger.error(f"Database error: {db_error}", exc_info=True)
            db.rollback()
            raise HTTPException(status_code=500, detail=f"Database error: {str(db_error)}")

    finally:
        db.close()


@app.get("/interactions/{interaction_id}")
def get_interaction(interaction_id: str):
    """Get a specific interaction by ID."""
    try:
        db = get_db_session()
        interaction = db.query(Interaction).filter_by(id=uuid.UUID(interaction_id)).first()

        if not interaction:
            raise HTTPException(status_code=404, detail="Interaction not found")

        db.close()

        return interaction
    except Exception as e:
        logger.error(f"Error fetching interaction: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to fetch interaction: {str(e)}")


@app.post("/interactions/{interaction_id}/inference")
def interaction_inference(interaction_id: str):
    """Inference endpoint with database-backed context integration."""
    db = get_db_session()

    try:
        interaction = db.query(Interaction).filter_by(id=uuid.UUID(interaction_id)).first()

        if not interaction:
            raise HTTPException(status_code=404, detail="Interaction not found")

        # Use database-backed context processing
        context, has_intent = context_processor.build_context(interaction)

        if not has_intent:
            return {"message": "Intent not recognized, no inference performed."}

        response = InferenceProcessor.send_prompt(prompt=interaction.text, context=context)  # type: ignore

        # Add context information to response with database queries
        response["context_used"] = str(bool(context))

        # Get enhanced features from the database interaction
        enhanced_features = {
            "entities": interaction.entities,
            "sentiment": interaction.sentiment,
            "speaker_id": str(interaction.speaker_id),
            "conversation_id": (
                str(interaction.conversation_id) if bool(interaction.conversation_id) else None
            ),
        }
        response["enhanced_features"] = json.dumps(enhanced_features)

        return response

    except Exception as e:
        logger.error(f"Error in inference: {e}")
        raise HTTPException(status_code=500, detail=f"Inference failed: {str(e)}")

    finally:
        db.close()


@app.get("/interactions", deprecated=True)
def get_interactions(limit: int = 0):
    """Get recent interactions for live interaction display."""
    try:
        db = get_db_session()
        query = db.query(Interaction).order_by(Interaction.timestamp.desc())
        if limit != 0:
            query = query.limit(limit)
        interactions = query.all()

        db.close()

        return interactions
    except Exception as e:
        logger.error(f"Error fetching interactions: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to fetch interactions: {str(e)}")


@app.delete("/interactions/{interaction_id}")
def delete_interaction(interaction_id: str):
    """Delete a specific interaction from the database."""
    try:
        db = get_db_session()
        try:
            interaction = db.query(Interaction).filter_by(id=uuid.UUID(interaction_id)).first()
            if not interaction:
                raise HTTPException(status_code=404, detail="Interaction not found")

            db.delete(interaction)
            db.commit()

            status["recent_interactions"].remove(interaction.id)

            return {"detail": "Interaction deleted successfully"}
        except Exception as e:
            db.rollback()
            raise e
        finally:
            db.close()
    except Exception as e:
        logger.error(f"Error deleting interaction: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to delete interaction: {str(e)}")


@app.get("/conversations")
def get_recent_conversations(limit: int = 10):
    """Get recent conversations with their interactions."""
    try:
        db = get_db_session()
        try:
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

        finally:
            db.close()

    except Exception as e:
        logger.error(f"Error fetching conversations: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to fetch conversations: {str(e)}")


@app.get("/person/{person_id}")
def get_person(person_id: str):
    """Get a specific person by ID."""
    try:
        db = get_db_session()
        try:
            person = db.query(Person).filter_by(id=uuid.UUID(person_id)).first()
            if not person:
                raise HTTPException(status_code=404, detail="Person not found")

            return person
        finally:
            db.close()
    except Exception as e:
        logger.error(f"Error fetching person: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to fetch person: {str(e)}")


@app.get("/streams/best")
def get_best_stream():
    """Get the currently selected best audio stream."""
    best_stream_info = audio_scorer.get_best_stream()

    if not best_stream_info:
        return {"best_stream": None}

    return {"best_stream": best_stream_info}


@app.get("/streams/scores")
def get_all_stream_scores():
    """Get quality scores for all active streams."""

    try:
        clients_info = audio_scorer.clients
        scores = audio_scorer.get_all_stream_scores()
        best_stream = audio_scorer.get_best_stream()

        return {
            "active_streams": len(clients_info),
            "stream_scores": scores,
            "current_best": best_stream,
        }

    except Exception as e:
        logger.error(f"Error getting stream scores: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get stream scores: {str(e)}")


@app.get("/streams/{client_id}/info")
def get_client_stream_info(client_id: str):
    """Get detailed stream information about a specific client."""

    if client_id not in audio_scorer.clients:
        raise HTTPException(
            status_code=404, detail=f"Client {client_id} not found in stream scoring"
        )

    client_info = audio_scorer.clients[client_id]
    current_score = audio_scorer.get_all_stream_scores().get(client_id, 0.0)
    best_stream = audio_scorer.get_best_stream()
    is_best_stream = best_stream and best_stream.get("client_id") == client_id

    return {
        "client_id": client_id,
        "quality_metrics": client_info.quality_metrics.__dict__,
        "current_score": round(current_score, 2),
        "is_best_stream": is_best_stream,
        "last_update": client_info.last_update,
    }


@app.get("/service/{client_id}")
def get_client_info(client_id: str):
    """Get detailed information about a specific client."""

    client_dict = status["connected_clients"].get(client_id)

    if not client_dict:
        raise HTTPException(status_code=404, detail=f"Client {client_id} not found")

    return client_dict


@app.post("/streams/phone/location")
def update_phone_location(request: dict = Body(...)):
    """Update GPS-based location data for phone tracking."""
    try:
        client_id = request.get("client_id")
        location = request.get("location")

        if not client_id:
            raise HTTPException(status_code=400, detail="client_id is required")
        if not location:
            raise HTTPException(status_code=400, detail="location data is required")

        # Validate location data structure
        required_fields = ["latitude", "longitude"]
        for field in required_fields:
            if field not in location:
                raise HTTPException(status_code=400, detail=f"location.{field} is required")

        # Update phone location in audio scorer
        success = audio_scorer.set_phone_location(client_id, location)

        if not success:
            raise HTTPException(status_code=404, detail=f"Client {client_id} not found")

        logger.info(f"Updated phone location for {client_id}: {location}")

        return {
            "message": f"Phone location updated successfully for {client_id}",
            "location": location,
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error updating phone location: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to update phone location: {str(e)}")


@app.post("/streams/phone/rssi")
def update_phone_rssi(request: dict = Body(...)):
    """Update RSSI-based proximity data for phone tracking to specific client."""
    try:
        phone_client_id = request.get("phone_client_id")  # The phone doing the measurement
        target_client_id = request.get("target_client_id")  # The client being measured
        rssi = request.get("rssi")

        if not phone_client_id:
            raise HTTPException(status_code=400, detail="phone_client_id is required")
        if not target_client_id:
            raise HTTPException(status_code=400, detail="target_client_id is required")
        if rssi is None:
            raise HTTPException(status_code=400, detail="rssi value is required")

        # Update RSSI between phone and target client
        success = audio_scorer.set_phone_rssi(target_client_id, rssi)

        if not success:
            raise HTTPException(
                status_code=404, detail=f"Target client {target_client_id} not found"
            )

        logger.info(f"Updated RSSI from {phone_client_id} to {target_client_id}: {rssi} dBm")

        return {
            "message": f"RSSI updated successfully from {phone_client_id} to {target_client_id}",
            "rssi": rssi,
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error updating phone RSSI: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to update phone RSSI: {str(e)}")
