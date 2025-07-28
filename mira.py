import uuid
import warnings
import os
from datetime import datetime, timedelta
from fastapi import Body, FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from models import Interaction, Person
from db import get_db_session
import uvicorn
import logging
import json
from difflib import SequenceMatcher

# Suppress webrtcvad deprecation warnings as early as possible
warnings.filterwarnings(
    "ignore", category=UserWarning, message="pkg_resources is deprecated as an API"
)

# Lazy loading variables to prevent duplicate initialization
_advanced_modules_loaded = False
_context_processor_initialized = False

def log_once(message, flag_name=None):
    """Log a message only once during initialization"""
    if flag_name == 'advanced' and not globals().get('_advanced_logged', False):
        print(message)
        globals()['_advanced_logged'] = True
    elif flag_name == 'context' and not globals().get('_context_logged', False):
        print(message)
        globals()['_context_logged'] = True
    elif flag_name is None and not globals().get('_general_logged', False):
        print(message)
        globals()['_general_logged'] = True

def load_advanced_features():
    """Load advanced features - required for Mira to function"""
    global _advanced_modules_loaded
    global send_prompt, transcribe_interaction, create_enhanced_context_processor, process_interaction_enhanced, DEFAULT_CONFIG
    
    if _advanced_modules_loaded:
        return
    
    try:
        from run_inference import send_prompt
        from sentence_processor import transcribe_interaction
        from enhanced_context_processor import create_enhanced_context_processor, process_interaction_enhanced
        from context_config import DEFAULT_CONFIG
        log_once("✅ Advanced AI features loaded successfully", 'advanced')
        _advanced_modules_loaded = True
    except ImportError as e:
        error_msg = f"❌ Failed to load required AI features: {e}"
        log_once(error_msg, 'advanced')
        log_once("💥 Mira requires advanced AI features to function properly.", 'advanced')
        log_once("Please ensure all dependencies are installed and try again.", 'advanced')
        raise SystemExit(f"CRITICAL ERROR: {error_msg}") from e

def initialize_context_processor():
    """Initialize context processor - required for Mira to function"""
    global _context_processor_initialized, context_processor
    
    # Ensure advanced features are loaded first
    load_advanced_features()
    
    if _context_processor_initialized:
        return
    
    try:
        context_processor = create_enhanced_context_processor(DEFAULT_CONFIG)
        log_once("✅ Enhanced context processor initialized", 'context')
        _context_processor_initialized = True
    except Exception as e:
        error_msg = f"❌ Failed to initialize context processor: {e}"
        log_once(error_msg, 'context')
        log_once("💥 Mira requires context processor to function properly.", 'context')
        raise SystemExit(f"CRITICAL ERROR: {error_msg}") from e

# Suppress webrtcvad deprecation warnings
warnings.filterwarnings(
    "ignore", category=UserWarning, message="pkg_resources is deprecated as an API"
)


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

def get_current_status():
    """Get current status - features should already be loaded at startup"""
    return status

status: dict = {
    "version": "2.3.9",  # Removed simple mode fallback
    "listening_clients": list(),
    "enabled": False,
    "mode": "advanced",  # Always advanced mode
    "features": {
        "advanced_nlp": True,           # Always available
        "speaker_clustering": True,     # Always available
        "context_summarization": True,  # Always available
        "database_integration": True,   # Always available
        "audio_processing": True        # Always available
    }
}


@app.get("/")
def root():
    return get_current_status()


@app.post("/register_client")
def register_client(client_id: str):
    status["listening_clients"].append(client_id)
    print("Client registered:", client_id)
    return status


@app.post("/deregister_client")
def deregister_client(client_id: str):
    if client_id in status["listening_clients"]:
        status["listening_clients"].remove(client_id)
    else:
        raise HTTPException(status_code=404, detail="Client not found")

    print("Client deregistered:", client_id)
    return status


@app.patch("/enable")
def enable_service():
    status["enabled"] = True
    print("Mira enabled")
    return status


@app.patch("/disable")
def disable_service():
    status["enabled"] = False
    print("Mira disabled")
    return status


def is_duplicate_transcription(text: str, speaker: int, similarity_threshold: float = 0.85, time_window_minutes: int = 2) -> bool:
    """Check if a transcription is a duplicate of a recent one."""
    try:
        db = get_db_session()
        try:
            # Get recent interactions from the same speaker within time window
            cutoff_time = datetime.utcnow() - timedelta(minutes=time_window_minutes)
            recent_interactions = db.query(Interaction).filter(
                Interaction.user_id == speaker,
                Interaction.timestamp >= cutoff_time
            ).all()
            
            # Check similarity with recent transcriptions
            for interaction in recent_interactions:
                similarity = SequenceMatcher(None, text.lower().strip(), interaction.text.lower().strip()).ratio()
                if similarity >= similarity_threshold:
                    logger.info(f"Duplicate transcription detected (similarity: {similarity:.2f}): '{text}' vs '{interaction.text}'")
                    return True
            
            return False
        finally:
            db.close()
    except Exception as e:
        logger.warning(f"Error checking for duplicate transcription: {e}")
        return False

@app.post("/process_interaction")
def process_interaction(sentence_buf_raw: bytes = Body(...)):
    """Process interaction - transcribe sentence and identify speaker."""

    try:
        if len(sentence_buf_raw) == 0:
            raise ValueError("No audio data received")
        
        logger.info(f"Processing audio data: {len(sentence_buf_raw)} bytes")
        
        # Use advanced processing (features should already be loaded)
        sentence_buf = bytearray(sentence_buf_raw)
        transcription_result = transcribe_interaction(sentence_buf)
        logger.info("Advanced transcription successful")
        
        # Check for duplicate transcription before saving
        if is_duplicate_transcription(transcription_result["text"], transcription_result["speaker"]):
            logger.info("Skipping duplicate transcription")
            return {"message": "Duplicate transcription skipped"}
        
        # Create database interaction
        interaction = Interaction(
            user_id=transcription_result["speaker"],
            text=transcription_result["text"]
        )

        # Check for shutdown command first
        if "mira" in interaction.text.lower() and any(
            cancelCMD in interaction.text.lower() for cancelCMD in ("cancel", "exit")
        ):
            logger.info("Mira interrupted by voice command")
            disable_service()
            return status

        # Save to database with better error handling
        db = get_db_session()
        try:
            db.add(interaction)
            db.commit()
            db.refresh(interaction)
            logger.info(f"Interaction saved to database with ID: {interaction.id}")
            
            # Return interaction data for the frontend
            result = {
                "id": str(interaction.id),
                "user_id": interaction.user_id,
                "speaker": interaction.user_id,
                "text": interaction.text,
                "timestamp": interaction.timestamp.isoformat() if getattr(interaction, "timestamp") else None
            }
            
            logger.info(f"Returning interaction data: {result}")
            return result
            
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
def get_recent_interactions(limit: int = 10):
    """Get recent interactions for live transcription display."""
    try:
        db = get_db_session()
        try:
            interactions = db.query(Interaction).order_by(
                Interaction.timestamp.desc()
            ).limit(limit).all()
            
            result = []
            for interaction in interactions:
                result.append({
                    "id": str(interaction.id),
                    "user_id": interaction.user_id,
                    "speaker": interaction.user_id,
                    "text": interaction.text,
                    "timestamp": interaction.timestamp.isoformat() if getattr(interaction, "timestamp") else None
                })
            
            return result
        finally:
            db.close()
    except Exception as e:
        logger.error(f"Error fetching interactions: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to fetch interactions: {str(e)}")


@app.delete("/interactions")
def clear_all_interactions():
    """Clear all interactions from the database."""
    try:
        db = get_db_session()
        try:
            # Delete all interactions
            deleted_count = db.query(Interaction).delete()
            db.commit()
            
            logger.info(f"Cleared {deleted_count} interactions from database")
            
            # Also clear from context processor if initialized
            if _context_processor_initialized and 'context_processor' in globals():
                context_processor.interaction_history.clear()
                logger.info("Cleared interactions from context processor")
            
            return {
                "message": f"Successfully cleared {deleted_count} interactions",
                "deleted_count": deleted_count
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
    """Enhanced inference endpoint with context integration."""
    try:
        # Features should already be loaded at startup
        interaction = (
            get_db_session().query(Interaction).filter_by(id=uuid.UUID(interaction_id)).first()
        )

        if not interaction:
            raise HTTPException(status_code=404, detail="Interaction not found")

        # Extract voice embedding if available from transcription
        voice_embedding = None
        
        # Use enhanced context processing
        
        initialize_context_processor()
        context, has_intent = process_interaction_enhanced(
            context_processor, 
            interaction, 
            voice_embedding
        )

        if not has_intent:
            return {"message": "Intent not recognized, no inference performed."}

        # Send enhanced prompt with context
        enhanced_prompt = f"{str(interaction.text)}\n\nContext:\n{context}" if context else str(interaction.text)
        response = send_prompt(prompt=enhanced_prompt, context=context)
        
        # Add context information to response
        response["context_used"] = str(bool(context))
        response["enhanced_features"] = json.dumps({
            "entities": getattr(context_processor.interaction_history[-1], 'entities', None) if context_processor and context_processor.interaction_history else None,
            "sentiment": getattr(context_processor.interaction_history[-1], 'sentiment', None) if context_processor and context_processor.interaction_history else None
        })
        
        return response
        
    except Exception as e:
        logger.error(f"Error in inference: {e}")
        raise HTTPException(status_code=500, detail=f"Inference failed: {str(e)}")


@app.get("/context/speakers")
def get_speaker_summary():
    """Get summary of all tracked speakers."""
    initialize_context_processor()
    return context_processor.get_speaker_summary()


@app.get("/context/history")
def get_interaction_history(limit: int = 10):
    """Get recent interaction history."""
    try:
        initialize_context_processor()
        recent_interactions = context_processor.interaction_history[-limit:]
        return [interaction.to_dict() for interaction in recent_interactions]
    except Exception as e:
        # Fallback to database query if context processor fails
        logger.warning(f"Context processor unavailable, falling back to database: {e}")
        try:
            db = get_db_session()
            try:
                interactions = db.query(Interaction).order_by(
                    Interaction.timestamp.desc()
                ).limit(limit).all()
                
                return [
                    {
                        "id": str(interaction.id),
                        "speaker": interaction.user_id,
                        "text": interaction.text,
                        "timestamp": interaction.timestamp.isoformat() if getattr("interaction", "timestamp") else None
                    }
                    for interaction in interactions
                ]
            finally:
                db.close()
        except Exception as db_error:
            logger.error(f"Error fetching interaction history from database: {db_error}")
            raise HTTPException(status_code=500, detail=f"Failed to fetch interaction history: {str(db_error)}")


@app.post("/context/identify_speaker")
def identify_speaker(speaker_index: int, name: str):
    """Manually identify a speaker."""
    try:
        db = get_db_session()
        try:
            person = db.query(Person).filter_by(speaker_index=speaker_index).first()
            
            if person:
                setattr(person, "name", name)
                setattr(person, "is_identified", True)
                
                db.commit()
                
                # Update context processor
                initialize_context_processor()
                if speaker_index in context_processor.speaker_profiles:
                    context_processor.speaker_profiles[speaker_index].name = name
                    context_processor.speaker_profiles[speaker_index].is_identified = True
                    
                return {"message": f"Speaker {speaker_index} identified as {name}"}
            else:
                raise HTTPException(status_code=404, detail="Speaker not found")
        finally:
            db.close()
            
    except Exception as e:
        logger.error(f"Error identifying speaker: {e}")
        raise HTTPException(status_code=500, detail=f"Speaker identification failed: {str(e)}")


# Main entry point
if __name__ == "__main__":
    # Validate advanced features at startup to fail fast
    try:
        load_advanced_features()
        initialize_context_processor()
        print("✅ Mira backend initialized successfully with all required features")
    except SystemExit as e:
        print(f"\n{e}")
        print("⛔ Mira backend cannot start without required AI features.")
        print("Please install all dependencies and try again.")
        exit(1)
    
    # Use reload only in development, not when launched via script
    reload_mode = os.getenv("MIRA_DEV_MODE", "false").lower() == "true"
    uvicorn.run("mira:app", host="0.0.0.0", port=8000, reload=reload_mode)