import uuid
from fastapi import Body, FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from models import Interaction, Person
from db import get_db_session
import logging
import json
from run_inference import send_prompt
from sentence_processor import transcribe_interaction
from context_processor import (
    create_context_processor,
    process_interaction,
)
from context_config import DEFAULT_CONFIG
import numpy as np


# Initialize FastAPI app first
app = FastAPI()

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins for development
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods
    allow_headers=["*"],  # Allows all headers
)

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
context_processor = create_context_processor(DEFAULT_CONFIG)


status: dict = {
    "version": "2.3.9",  # Removed simple mode fallback
    "listening_clients": list(),
    "enabled": False,
    "mode": "advanced",  # Always advanced mode
    "features": {
        "advanced_nlp": True,  # Always available
        "speaker_clustering": True,  # Always available
        "context_summarization": True,  # Always available
        "database_integration": True,  # Always available
        "audio_processing": True,  # Always available
    },
}


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


@app.post("/register_client")
def register_client(client_id: str):
    status["listening_clients"].append(client_id)
    return status


@app.post("/deregister_client")
def deregister_client(client_id: str):
    if client_id in status["listening_clients"]:
        status["listening_clients"].remove(client_id)
        print("Client deregistered:", client_id)
    else:
        # Make endpoint idempotent - don't error if client already deregistered
        print("Client already deregistered or not found:", client_id)

    return status


@app.patch("/enable")
def enable_service():
    status["enabled"] = True
    print("Mira enabled")
    return status


@app.patch("/disable")
def disable_service():
    status["enabled"] = False
    return status


@app.post("/register_interaction")
def register_interaction(sentence_buf_raw: bytes = Body(...)):
    """Register interaction - transcribe sentence, identify speaker, and process context."""

    try:
        if len(sentence_buf_raw) == 0:
            raise ValueError("No audio data received")

        logger.info(f"Processing audio data: {len(sentence_buf_raw)} bytes")

        # Use advanced processing with integrated speaker recognition
        sentence_buf = bytearray(sentence_buf_raw)
        transcription_result = transcribe_interaction(sentence_buf)

        if transcription_result is None:
            return

        logger.info("Advanced transcription successful")

        # Extract voice embedding for speaker recognition
        voice_embedding = None
        if transcription_result.get('voice_embedding') is not None:
            voice_embedding = np.array(transcription_result['voice_embedding'])
            del transcription_result['voice_embedding']  # Remove from dict to avoid DB errors
        elif 'voice_embedding' in transcription_result:
            # Remove None voice_embedding to avoid errors
            del transcription_result['voice_embedding']

        # Check for shutdown command first
        if "mira" in transcription_result['text'].lower() and any(
            cancelCMD in transcription_result['text'].lower() for cancelCMD in ("cancel", "exit")
        ):
            logger.info("Mira interrupted by voice command")
            disable_service()
            return status

        # Use context processor for speaker recognition and database integration
        db = get_db_session()
        try:
            # Assign speaker using robust method from context processor
            if voice_embedding is not None:
                person_id = context_processor.assign_speaker(voice_embedding)
            else:
                # Check if a default person already exists
                person = db.query(Person).filter_by(speaker_index=1).first()
                if not person:
                    person = Person(speaker_index=1, name="Person 1")
                    db.add(person)
                    db.commit()
                    db.refresh(person)
                person_id = person.id

            # Create database interaction with speaker assignment
            interaction = Interaction(
                text=transcription_result['text'],
                speaker_id=person_id,
            )

            # Process interaction through context processor for conversation management
            if voice_embedding is not None:
                context_processor.update_speaker_clustering(voice_embedding, person_id)

            # Save to database
            db.add(interaction)
            db.commit()
            db.refresh(interaction)
            logger.info(f"Interaction saved to database with ID: {interaction.id}")

            return interaction

        except Exception as db_error:
            logger.error(f"Database error: {db_error}")
            db.rollback()
            raise HTTPException(status_code=500, detail=f"Database error: {str(db_error)}")
        finally:
            db.close()

    except Exception as e:
        logger.error(f"Error processing interaction: {e}")
        raise HTTPException(status_code=500, detail=f"Processing failed: {str(e)}")


@app.get("/interactions")
def get_interactions(limit: int = 0):
    """Get recent interactions for live transcription display."""
    try:
        db = get_db_session()
        try:
            query = db.query(Interaction).order_by(Interaction.timestamp.desc())
            if limit != 0:
                query = query.limit(limit)
            interactions = query.all()

            return interactions
        finally:
            db.close()
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
                interactions_to_delete = db.query(Interaction.id).order_by(Interaction.timestamp.asc()).limit(limit).all()
                interaction_ids = [interaction.id for interaction in interactions_to_delete]
                deleted_count = db.query(Interaction).filter(Interaction.id.in_(interaction_ids)).delete(synchronize_session=False)
            else:
                # Delete all interactions
                deleted_count = db.query(Interaction).delete()

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


@app.post("/inference")
def inference_endpoint(interaction_id: str):
    """Inference endpoint with database-backed context integration."""
    try:
        db = get_db_session()
        try:
            interaction = db.query(Interaction).filter_by(id=uuid.UUID(interaction_id)).first()

            if not interaction:
                raise HTTPException(status_code=404, detail="Interaction not found")

            # Use database-backed context processing
            context, has_intent = process_interaction(context_processor, interaction)

            if not has_intent:
                return {"message": "Intent not recognized, no inference performed."}

            # Send prompt with context
            enhanced_prompt = (
                f"{str(interaction.text)}\n\nContext:\n{context}" if context else str(interaction.text)
            )
            response = send_prompt(prompt=enhanced_prompt, context=context)

            # Add context information to response with database queries
            response["context_used"] = str(bool(context))

            # Get enhanced features from the database interaction
            enhanced_features = {
                "entities": interaction.entities,
                "sentiment": interaction.sentiment,
                "speaker_id": str(interaction.speaker_id),
                "conversation_id": str(interaction.conversation_id) if bool(interaction.conversation_id) else None,
            }
            response["enhanced_features"] = json.dumps(enhanced_features)

            return response

        finally:
            db.close()

    except Exception as e:
        logger.error(f"Error in inference: {e}")
        raise HTTPException(status_code=500, detail=f"Inference failed: {str(e)}")


@app.get("/context/speakers")
def get_speaker_summary():
    """Get summary of all tracked speakers from database."""
    return context_processor.get_speaker_summary()


@app.get("/context/conversations")
def get_recent_conversations(limit: int = 10):
    """Get recent conversations with their interactions."""
    try:
        db = get_db_session()
        try:
            from models import Conversation
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
                    "end_time": conv.end_of_conversation.isoformat() if conv.end_of_conversation is not None else None,
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
