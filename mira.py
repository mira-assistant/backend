import uuid
from fastapi import Body, FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import logging
import json
from collections import deque
from contextlib import asynccontextmanager


from db import get_db_session
from models import Interaction, Person, Conversation


import inference_processor
import sentence_processor
import context_processor
from audio_stream_scorer import AudioStreamScorer
from wake_word_detector import WakeWordDetector


# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
processor = context_processor.create_context_processor()

# Initialize audio stream scorer
audio_scorer = AudioStreamScorer()

# Initialize wake word detector
wake_word_detector = WakeWordDetector()

status: dict = {
    "version": "4.1.1",
    "listening_clients": list(),
    "enabled": False,
    "mode": "advanced",
    "features": {
        "advanced_nlp": True,
        "speaker_clustering": True,
        "context_summarization": True,
        "database_integration": True,
        "audio_processing": True,
        "stream_scoring": True,
        "wake_word_detection": True,  # New feature
    },
    "recent_interactions": deque(maxlen=10),  # Use deque as a queue with a max size
    "current_best_stream": None,  # Track the best performing stream
}


@asynccontextmanager
async def lifespan(app: FastAPI):
    for interaction in (
        get_db_session().query(Interaction).order_by(Interaction.timestamp.desc()).limit(10).all()
    ):
        status["recent_interactions"].append(interaction.id)
    yield


hosting_urls = {
    "localhost": "http://localhost:8000",
    "ankurs-macbook-air": "http://100.75.140.79:8000",
}

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
    return status


@app.post("/service/client/register/{client_id}")
def register_client(client_id: str, device_type: str = None, location: dict = None):
    """Register a client and initialize stream scoring."""
    status["listening_clients"].append(client_id)

    # Register client with audio stream scorer
    success = audio_scorer.register_client(
        client_id=client_id,
        device_type=device_type,
        location=location
    )

    if success:
        logger.info(f"Client {client_id} registered for stream scoring")

    return {"message": f"{client_id} registered successfully", "stream_scoring_enabled": success}


@app.delete("/service/client/deregister/{client_id}")
def deregister_client(client_id: str):
    """Deregister a client and remove from stream scoring."""
    if client_id in status["listening_clients"]:
        status["listening_clients"].remove(client_id)
    else:
        print("Client already deregistered or not found:", client_id)

    # Deregister from audio stream scorer
    success = audio_scorer.deregister_client(client_id)

    # Update best stream if this client was selected
    best_stream_info = audio_scorer.get_best_stream()
    status["current_best_stream"] = best_stream_info[0] if best_stream_info else None

    if client_id not in status["listening_clients"] and not success:
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
def register_interaction(sentence_buf_raw: bytes = Body(...), client_id: str = None):
    """Register interaction - transcribe sentence, identify speaker, and update stream quality."""

    try:
        if len(sentence_buf_raw) == 0:
            raise ValueError("No audio data received")

        logger.info(f"Processing audio data: {len(sentence_buf_raw)} bytes from client: {client_id}")

        # Step 1: Transcribe and get voice embedding (no NLP)
        sentence_buf = bytearray(sentence_buf_raw)
        transcription_result = sentence_processor.transcribe_interaction(sentence_buf)

        if transcription_result is None:
            return

        logger.info("Advanced transcription successful")

        # Step 1.5: Update stream quality if client_id provided
        if client_id:
            # Convert audio for quality analysis
            audio_float = sentence_processor.pcm_bytes_to_float32(sentence_buf_raw)
            metrics = audio_scorer.update_stream_quality(client_id, audio_float)

            if metrics:
                logger.debug(f"Updated stream quality for {client_id}: SNR={metrics.snr:.1f}dB")

            # Update best stream selection
            best_stream_info = audio_scorer.get_best_stream()
            status["current_best_stream"] = best_stream_info[0] if best_stream_info else None

        # Step 1.6: Check for wake words in transcribed text
        if transcription_result and transcription_result.get("text"):
            # Calculate audio length from bytes (assuming 16kHz, 16-bit, mono PCM)
            audio_length = len(sentence_buf_raw) / (16000 * 2)  # bytes / (sample_rate * bytes_per_sample)
            
            wake_word_detection = wake_word_detector.process_audio_text(
                client_id=client_id or "unknown",
                transcribed_text=transcription_result["text"],
                audio_length=audio_length
            )
            
            if wake_word_detection:
                logger.info(f"Wake word '{wake_word_detection.wake_word}' detected from client {client_id}")
                # Wake word detected - could trigger specific actions here
                # For now, we just log it, but this could be extended to:
                # - Start/stop recording
                # - Change system state
                # - Send notifications
                # - Trigger specific workflows

        # Step 2: Check for shutdown command
        if "mira" in transcription_result["text"].lower() and any(
            cancelCMD in transcription_result["text"].lower() for cancelCMD in ("cancel", "exit")
        ):
            logger.info("Mira interrupted by voice command")
            disable_service()
            return {"message": "Service disabled by voice command"}

        # Step 3: Assign speaker, create interaction, save basic info
        db = get_db_session()
        try:
            interaction = Interaction(
                **transcription_result,
            )
            db.add(interaction)
            db.commit()
            db.refresh(interaction)

            speaker_id = processor.assign_speaker(
                transcription_result["voice_embedding"],
                session=db,
                interaction_id=interaction.id,  # Pass interaction_id for advanced cache
            )

            interaction.speaker_id = speaker_id

            db.commit()
            db.refresh(interaction)

            logger.info(f"Interaction saved to database with ID: {interaction.id}")
            status["recent_interactions"].append(interaction.id)

            # Return minimal interaction details for frontend display
            response = {
                "id": str(interaction.id),
                "text": interaction.text,
                "timestamp": interaction.timestamp.isoformat(),
                "speaker_id": str(interaction.speaker_id),
            }

            # Include stream quality info if available
            if client_id and client_id in [client.client_id for client in audio_scorer.clients.values()]:
                response["stream_quality"] = {
                    "client_id": client_id,
                    "is_best_stream": client_id == status["current_best_stream"]
                }

            return response

        except Exception as db_error:
            logger.error(f"Database error: {db_error}", exc_info=True)
            db.rollback()
            raise HTTPException(status_code=500, detail=f"Database error: {str(db_error)}")
        finally:
            db.close()

    except Exception as e:
        logger.error(f"Error processing interaction: {e}")
        raise HTTPException(status_code=500, detail=f"Processing failed: {str(e)}")


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


@app.post("/interactions/process/{interaction_id}")
def inference_endpoint(interaction_id: str):
    """Inference endpoint with database-backed context integration."""
    try:
        db = get_db_session()
        interaction = db.query(Interaction).filter_by(id=uuid.UUID(interaction_id)).first()

        if not interaction:
            raise HTTPException(status_code=404, detail="Interaction not found")

        # Use database-backed context processing
        context, has_intent = context_processor.process_interaction(processor, interaction)

        if not has_intent:
            return {"message": "Intent not recognized, no inference performed."}

        # Send prompt with context
        enhanced_prompt = (
            f"{str(interaction.text)}\n\nContext:\n{context}" if context else str(interaction.text)
        )
        response = inference_processor.send_prompt(prompt=enhanced_prompt, context=context)

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

        db.close()

        return response

    except Exception as e:
        logger.error(f"Error in inference: {e}")
        raise HTTPException(status_code=500, detail=f"Inference failed: {str(e)}")


@app.get("/interactions", deprecated=True)
def get_interactions(limit: int = 0):
    """Get recent interactions for live transcription display."""
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


@app.delete("/interactions")
def delete_interactions(limit: int = 0):
    """Clear all interactions from the database."""
    try:
        db = get_db_session()
        try:
            if limit != 0:
                # If limit is specified, first get the IDs to delete
                interactions_to_delete = (
                    db.query(Interaction.id)
                    .order_by(Interaction.timestamp.asc())
                    .limit(limit)
                    .all()
                )
                interaction_ids = [interaction.id for interaction in interactions_to_delete]
                deleted_count = (
                    db.query(Interaction)
                    .filter(Interaction.id.in_(interaction_ids))
                    .delete(synchronize_session=False)
                )

                status["recent_interactions"] = deque(
                    [i for i in status["recent_interactions"] if i not in interaction_ids],
                    maxlen=10,
                )

            else:
                # Delete all interactions
                deleted_count = db.query(Interaction).delete()
                status["recent_interactions"].clear()

            db.commit()

            logger.info(f"Cleared {deleted_count} interactions from database")

            return {
                "deleted_count": deleted_count,
            }
        except Exception as e:
            db.rollback()
            raise e
        finally:
            db.close()
    except Exception as e:
        logger.error(f"Error clearing interactions: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to clear interactions: {str(e)}")


@app.get("/conversations")
def get_recent_conversations(limit: int = 10):
    """Get recent conversations with their interactions."""
    try:
        db = get_db_session()
        try:
            conversations = (
                db.query(Conversation)
                .order_by(Conversation.start_of_conversation.desc())
                .limit(limit)
                .all()
            )

            result = []
            for conv in conversations:
                conv_data = {
                    "id": str(conv.id),
                    "start_time": conv.start_of_conversation.isoformat(),
                    "end_time": (
                        conv.end_of_conversation.isoformat()
                        if conv.end_of_conversation is not None
                        else None
                    ),
                    "participants": conv.participants,
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


@app.get("/speakers/{speaker_id}")
def get_speaker(speaker_id: str):
    """Get a specific speaker by ID."""
    try:
        db = get_db_session()
        try:
            speaker = db.query(Person).filter_by(id=uuid.UUID(speaker_id)).first()
            if not speaker:
                raise HTTPException(status_code=404, detail="Speaker not found")

            return speaker
        finally:
            db.close()
    except Exception as e:
        logger.error(f"Error fetching speaker: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to fetch speaker: {str(e)}")


# ============ Audio Stream Scoring Endpoints ============

@app.get("/streams/best")
def get_best_stream():
    """Get the currently selected best audio stream."""
    try:
        best_stream_info = audio_scorer.get_best_stream()

        if not best_stream_info:
            return {"message": "No active streams available", "best_stream": None}

        client_id, score = best_stream_info
        client_info = audio_scorer.get_client_info(client_id)

        response = {
            "best_stream": {
                "client_id": client_id,
                "score": round(score, 2),
                "metrics": {
                    "snr": round(client_info.quality_metrics.snr, 2),
                    "speech_clarity": round(client_info.quality_metrics.speech_clarity, 2),
                    "volume_level": round(client_info.quality_metrics.volume_level, 4),
                    "noise_level": round(client_info.quality_metrics.noise_level, 4),
                    "phone_distance": client_info.quality_metrics.phone_distance,
                    "last_update": client_info.last_update.isoformat(),
                    "sample_count": client_info.quality_metrics.sample_count
                }
            }
        }

        return response

    except Exception as e:
        logger.error(f"Error getting best stream: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get best stream: {str(e)}")


@app.get("/streams/scores")
def get_all_stream_scores():
    """Get quality scores for all active streams."""
    try:
        scores = audio_scorer.get_all_stream_scores()

        detailed_scores = {}
        for client_id, score in scores.items():
            client_info = audio_scorer.get_client_info(client_id)
            if client_info:
                detailed_scores[client_id] = {
                    "score": round(score, 2),
                    "metrics": {
                        "snr": round(client_info.quality_metrics.snr, 2),
                        "speech_clarity": round(client_info.quality_metrics.speech_clarity, 2),
                        "volume_level": round(client_info.quality_metrics.volume_level, 4),
                        "noise_level": round(client_info.quality_metrics.noise_level, 4),
                        "phone_distance": client_info.quality_metrics.phone_distance,
                        "last_update": client_info.last_update.isoformat(),
                        "sample_count": client_info.quality_metrics.sample_count,
                        "is_active": client_info.is_active
                    }
                }

        return {
            "active_streams": len(detailed_scores),
            "stream_scores": detailed_scores,
            "current_best": status["current_best_stream"]
        }

    except Exception as e:
        logger.error(f"Error getting stream scores: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get stream scores: {str(e)}")


@app.post("/streams/{client_id}/distance")
def set_phone_distance(client_id: str, request: dict = Body(...)):
    """Set phone distance for a client (future feature)."""
    try:
        distance = request.get("distance")
        if distance is None:
            raise HTTPException(status_code=400, detail="Distance value is required")

        success = audio_scorer.set_phone_distance(client_id, distance)

        if not success:
            raise HTTPException(status_code=404, detail=f"Client {client_id} not found")

        # Update best stream selection after distance change
        best_stream_info = audio_scorer.get_best_stream()
        status["current_best_stream"] = best_stream_info[0] if best_stream_info else None

        return {
            "message": f"Phone distance set for {client_id}",
            "distance": distance,
            "current_best_stream": status["current_best_stream"]
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error setting phone distance: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to set phone distance: {str(e)}")


@app.get("/streams/{client_id}/info")
def get_client_stream_info(client_id: str):
    """Get detailed information about a specific client's stream."""
    try:
        client_info = audio_scorer.get_client_info(client_id)

        if not client_info:
            raise HTTPException(status_code=404, detail=f"Client {client_id} not found")

        # Calculate current score
        current_score = audio_scorer.calculate_overall_score(client_info.quality_metrics)

        return {
            "client_id": client_info.client_id,
            "is_active": client_info.is_active,
            "device_type": client_info.device_type,
            "location": client_info.location,
            "last_update": client_info.last_update.isoformat(),
            "current_score": round(current_score, 2),
            "is_best_stream": client_id == status["current_best_stream"],
            "quality_metrics": {
                "snr": round(client_info.quality_metrics.snr, 2),
                "speech_clarity": round(client_info.quality_metrics.speech_clarity, 2),
                "volume_level": round(client_info.quality_metrics.volume_level, 4),
                "noise_level": round(client_info.quality_metrics.noise_level, 4),
                "phone_distance": client_info.quality_metrics.phone_distance,
                "sample_count": client_info.quality_metrics.sample_count,
                "timestamp": client_info.quality_metrics.timestamp.isoformat()
            }
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting client stream info: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get client info: {str(e)}")


@app.post("/streams/cleanup")
def cleanup_inactive_streams(timeout_seconds: int = 300):
    """Remove inactive streams that haven't been updated recently."""
    try:
        removed_clients = audio_scorer.cleanup_inactive_clients(timeout_seconds)

        # Also remove from listening_clients list
        for client_id in removed_clients:
            if client_id in status["listening_clients"]:
                status["listening_clients"].remove(client_id)

        # Update best stream selection
        best_stream_info = audio_scorer.get_best_stream()
        status["current_best_stream"] = best_stream_info[0] if best_stream_info else None

        return {
            "message": f"Cleaned up {len(removed_clients)} inactive streams",
            "removed_clients": removed_clients,
            "timeout_seconds": timeout_seconds,
            "current_best_stream": status["current_best_stream"]
        }

    except Exception as e:
        logger.error(f"Error cleaning up streams: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to cleanup streams: {str(e)}")


# ============ Wake Word Detection Endpoints ============

@app.get("/wake-words")
async def get_wake_words():
    """Get all configured wake words"""
    try:
        wake_words = wake_word_detector.get_wake_words()
        stats = wake_word_detector.get_stats()
        
        return {
            "wake_words": {
                word: {
                    "word": config.word,
                    "sensitivity": config.sensitivity,
                    "enabled": config.enabled,
                    "min_confidence": config.min_confidence,
                    "cooldown_seconds": config.cooldown_seconds
                }
                for word, config in wake_words.items()
            },
            "stats": stats
        }
    except Exception as e:
        logger.error(f"Error getting wake words: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get wake words: {str(e)}")


@app.post("/wake-words")
async def add_wake_word(
    wake_word_data: dict = Body(...)
):
    """Add a new wake word"""
    try:
        word = wake_word_data.get("word", "").strip()
        sensitivity = wake_word_data.get("sensitivity", 0.7)
        min_confidence = wake_word_data.get("min_confidence", 0.5)
        cooldown_seconds = wake_word_data.get("cooldown_seconds", 2.0)
        
        if not word:
            raise HTTPException(status_code=400, detail="Wake word cannot be empty")
        
        success = wake_word_detector.add_wake_word(
            word=word,
            sensitivity=sensitivity,
            min_confidence=min_confidence,
            cooldown_seconds=cooldown_seconds
        )
        
        if success:
            return {
                "message": f"Wake word '{word}' added successfully",
                "word": word,
                "sensitivity": sensitivity,
                "min_confidence": min_confidence,
                "cooldown_seconds": cooldown_seconds
            }
        else:
            raise HTTPException(status_code=400, detail="Failed to add wake word")
            
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error adding wake word: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to add wake word: {str(e)}")


@app.get("/wake-words/detections")
async def get_recent_detections(limit: int = 10):
    """Get recent wake word detections"""
    try:
        detections = wake_word_detector.get_recent_detections(limit)
        
        return {
            "detections": [
                {
                    "wake_word": detection.wake_word,
                    "confidence": detection.confidence,
                    "client_id": detection.client_id,
                    "timestamp": detection.timestamp.isoformat(),
                    "audio_snippet_length": detection.audio_snippet_length
                }
                for detection in detections
            ],
            "count": len(detections),
            "limit": limit
        }
    except Exception as e:
        logger.error(f"Error getting wake word detections: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get detections: {str(e)}")


@app.delete("/wake-words/detections")
async def clear_detections():
    """Clear all wake word detections and reset cooldowns"""
    try:
        wake_word_detector.clear_detections()
        return {
            "message": "All wake word detections cleared",
            "cleared": True
        }
    except Exception as e:
        logger.error(f"Error clearing wake word detections: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to clear detections: {str(e)}")


@app.patch("/wake-words/enable")
async def enable_wake_word_detection():
    """Enable the entire wake word detection system"""
    try:
        wake_word_detector.set_enabled(True)
        return {
            "message": "Wake word detection enabled",
            "enabled": True
        }
    except Exception as e:
        logger.error(f"Error enabling wake word detection: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to enable wake word detection: {str(e)}")


@app.patch("/wake-words/disable")
async def disable_wake_word_detection():
    """Disable the entire wake word detection system"""
    try:
        wake_word_detector.set_enabled(False)
        return {
            "message": "Wake word detection disabled",
            "enabled": False
        }
    except Exception as e:
        logger.error(f"Error disabling wake word detection: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to disable wake word detection: {str(e)}")


@app.post("/wake-words/process")
async def process_text_for_wake_words(
    data: dict = Body(...)
):
    """Process text for wake word detection (for testing or external integration)"""
    try:
        client_id = data.get("client_id", "unknown")
        text = data.get("text", "")
        audio_length = data.get("audio_length", 0.0)
        
        if not text:
            raise HTTPException(status_code=400, detail="Text cannot be empty")
        
        detection = wake_word_detector.process_audio_text(client_id, text, audio_length)
        
        if detection:
            return {
                "detected": True,
                "wake_word": detection.wake_word,
                "confidence": detection.confidence,
                "client_id": detection.client_id,
                "timestamp": detection.timestamp.isoformat(),
                "audio_snippet_length": detection.audio_snippet_length
            }
        else:
            return {
                "detected": False,
                "message": "No wake word detected in the provided text"
            }
            
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error processing text for wake words: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to process text: {str(e)}")


@app.delete("/wake-words/{word}")
async def remove_wake_word(word: str):
    """Remove a wake word"""
    try:
        success = wake_word_detector.remove_wake_word(word)
        
        if success:
            return {
                "message": f"Wake word '{word}' removed successfully",
                "word": word
            }
        else:
            raise HTTPException(status_code=404, detail=f"Wake word '{word}' not found")
            
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error removing wake word: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to remove wake word: {str(e)}")


@app.patch("/wake-words/{word}/enable")
async def enable_wake_word(word: str):
    """Enable a wake word"""
    try:
        success = wake_word_detector.set_wake_word_enabled(word, True)
        
        if success:
            return {
                "message": f"Wake word '{word}' enabled",
                "word": word,
                "enabled": True
            }
        else:
            raise HTTPException(status_code=404, detail=f"Wake word '{word}' not found")
            
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error enabling wake word: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to enable wake word: {str(e)}")


@app.patch("/wake-words/{word}/disable")
async def disable_wake_word(word: str):
    """Disable a wake word"""
    try:
        success = wake_word_detector.set_wake_word_enabled(word, False)
        
        if success:
            return {
                "message": f"Wake word '{word}' disabled",
                "word": word,
                "enabled": False
            }
        else:
            raise HTTPException(status_code=404, detail=f"Wake word '{word}' not found")
            
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error disabling wake word: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to disable wake word: {str(e)}")
