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


# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
processor = context_processor.create_context_processor()

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
    },
    "recent_interactions": deque(maxlen=10),  # Use deque as a queue with a max size
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
def register_client(client_id: str):
    status["listening_clients"].append(client_id)
    return {"message": f"{client_id} registered successfully"}


@app.delete("/service/client/deregister/{client_id}")
def deregister_client(client_id: str):
    if client_id in status["listening_clients"]:
        status["listening_clients"].remove(client_id)
    else:
        print("Client already deregistered or not found:", client_id)
        return {"message": f"{client_id} already deregistered or not found"}

    return {"message": f"{client_id} deregistered successfully"}


@app.patch("/service/enable")
def enable_service():
    status["enabled"] = True
    return {"message": "Service enabled successfully"}


@app.patch("/service/disable")
def disable_service():
    status["enabled"] = False
    return {"message": "Service disabled successfully"}


@app.post("/interactions/register")
def register_interaction(sentence_buf_raw: bytes = Body(...)):
    """Register interaction - transcribe sentence, identify speaker."""

    try:
        if len(sentence_buf_raw) == 0:
            raise ValueError("No audio data received")

        logger.info(f"Processing audio data: {len(sentence_buf_raw)} bytes")

        # Step 1: Transcribe and get voice embedding (no NLP)
        sentence_buf = bytearray(sentence_buf_raw)
        transcription_result = sentence_processor.transcribe_interaction(sentence_buf)

        if transcription_result is None:
            return

        logger.info("Advanced transcription successful")

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
            return {
                "id": str(interaction.id),
                "text": interaction.text,
                "timestamp": interaction.timestamp.isoformat(),
                "speaker_id": str(interaction.speaker_id),
            }

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
