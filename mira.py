import uuid
import warnings
import os
from fastapi import Body, FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from models import Interaction, Person
from db import get_db_session
import uvicorn
import logging
import json

# Suppress webrtcvad deprecation warnings as early as possible
warnings.filterwarnings(
    "ignore", category=UserWarning, message="pkg_resources is deprecated as an API"
)

# Only show initialization messages once
_INITIALIZATION_DONE = False

def log_once(message):
    """Log a message only once during initialization"""
    global _INITIALIZATION_DONE
    if not _INITIALIZATION_DONE:
        print(message)

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
    elif flag_name is None and not _INITIALIZATION_DONE:
        print(message)

def load_advanced_features():
    """Lazily load advanced features only when needed"""
    global _advanced_modules_loaded, ADVANCED_FEATURES_AVAILABLE
    global send_prompt, transcribe_interaction, create_enhanced_context_processor, process_interaction_enhanced, DEFAULT_CONFIG
    
    if _advanced_modules_loaded:
        return ADVANCED_FEATURES_AVAILABLE
    
    try:
        from run_inference import send_prompt
        from sentence_processor import transcribe_interaction
        from enhanced_context_processor import create_enhanced_context_processor, process_interaction_enhanced
        from context_config import DEFAULT_CONFIG
        ADVANCED_FEATURES_AVAILABLE = True
        log_once("✅ Advanced AI features loaded successfully", 'advanced')
        _advanced_modules_loaded = True
        return True
    except ImportError as e:
        log_once(f"⚠️  Advanced AI features unavailable: {e}", 'advanced')
        log_once("🔄 Falling back to simple processing mode", 'advanced')
        from simple_audio_processor import process_audio_simple
        globals()['process_audio_simple'] = process_audio_simple
        ADVANCED_FEATURES_AVAILABLE = False
        _advanced_modules_loaded = True
        return False

def initialize_context_processor():
    """Lazily initialize context processor only when needed"""
    global _context_processor_initialized, context_processor
    
    if _context_processor_initialized:
        return context_processor
    
    if not load_advanced_features():
        context_processor = None
        _context_processor_initialized = True
        return None
    
    try:
        context_processor = create_enhanced_context_processor(DEFAULT_CONFIG)
        log_once("✅ Enhanced context processor initialized", 'context')
        _context_processor_initialized = True
        return context_processor
    except Exception as e:
        log_once(f"⚠️  Enhanced context processor failed to initialize: {e}", 'context')
        context_processor = None
        _context_processor_initialized = True
        return None

# Initialize with defaults
ADVANCED_FEATURES_AVAILABLE = False
context_processor = None

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

# Initialize enhanced context processor only if advanced features are available
context_processor = initialize_context_processor()

# Update ADVANCED_FEATURES_AVAILABLE after lazy loading
if not _advanced_modules_loaded:
    load_advanced_features()

# Mark initialization as complete
_INITIALIZATION_DONE = True

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def get_current_status():
    """Get current status with updated feature availability"""
    load_advanced_features()  # Ensure features are checked
    return {
        "version": "2.3.7",  # Fixed duplicate initialization messages
        "listening_clients": status.get("listening_clients", []),
        "enabled": status.get("enabled", False),
        "mode": "advanced" if ADVANCED_FEATURES_AVAILABLE else "simple",
        "features": {
            "advanced_nlp": ADVANCED_FEATURES_AVAILABLE,
            "speaker_clustering": ADVANCED_FEATURES_AVAILABLE,
            "context_summarization": ADVANCED_FEATURES_AVAILABLE,
            "database_integration": True,  # Always available
            "audio_processing": True      # Always available
        }
    }

status: dict = {
    "version": "2.3.7",  # Fixed duplicate initialization messages
    "listening_clients": list(),
    "enabled": False,
    "mode": "simple",  # Will be updated by get_current_status
    "features": {
        "advanced_nlp": False,  # Will be updated by get_current_status
        "speaker_clustering": False,
        "context_summarization": False,
        "database_integration": True,  # Always available
        "audio_processing": True      # Always available
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


@app.post("/process_interaction")
def process_interaction(sentence_buf_raw: bytes = Body(...)):
    """Process interaction - transcribe sentence and identify speaker."""

    try:
        if len(sentence_buf_raw) == 0:
            raise ValueError("No audio data received")
        
        logger.info(f"Processing audio data: {len(sentence_buf_raw)} bytes")
        
        # Try advanced processing first, fall back to simple processing
        load_advanced_features()  # Ensure features are loaded
        if ADVANCED_FEATURES_AVAILABLE:
            try:
                sentence_buf = bytearray(sentence_buf_raw)
                transcription_result = transcribe_interaction(sentence_buf)
                logger.info("Advanced transcription successful")
            except Exception as e:
                logger.warning(f"Advanced transcription failed: {e}, falling back to simple processing")
                transcription_result = process_audio_simple(sentence_buf_raw)
        else:
            transcription_result = process_audio_simple(sentence_buf_raw)
            logger.info("Simple transcription completed")
        
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
                "timestamp": interaction.timestamp.isoformat() if interaction.timestamp else None
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
                    "timestamp": interaction.timestamp.isoformat() if interaction.timestamp else None
                })
            
            return result
        finally:
            db.close()
    except Exception as e:
        logger.error(f"Error fetching interactions: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to fetch interactions: {str(e)}")


@app.post("/inference")
def inference_endpoint(interaction_id: str):
    """Enhanced inference endpoint with context integration."""
    try:
        load_advanced_features()  # Ensure features are loaded
        if not ADVANCED_FEATURES_AVAILABLE:
            return {"message": "Advanced inference features not available in simple mode"}
        
        interaction = (
            get_db_session().query(Interaction).filter_by(id=uuid.UUID(interaction_id)).first()
        )

        if not interaction:
            raise HTTPException(status_code=404, detail="Interaction not found")

        # Extract voice embedding if available from transcription
        voice_embedding = None
        
        # Use enhanced context processing with the new function
        context_processor = initialize_context_processor()
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
    load_advanced_features()
    context_processor = initialize_context_processor()
    if not ADVANCED_FEATURES_AVAILABLE or not context_processor:
        return {"message": "Speaker tracking not available in simple mode"}
    return context_processor.get_speaker_summary()


@app.get("/context/history")
def get_interaction_history(limit: int = 10):
    """Get recent interaction history."""
    load_advanced_features()
    context_processor = initialize_context_processor()
    if not ADVANCED_FEATURES_AVAILABLE or not context_processor:
        # Fall back to database query
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
                        "timestamp": interaction.timestamp.isoformat() if interaction.timestamp else None
                    }
                    for interaction in interactions
                ]
            finally:
                db.close()
        except Exception as e:
            logger.error(f"Error fetching interaction history: {e}")
            return []
    
    recent_interactions = context_processor.interaction_history[-limit:]
    return [interaction.to_dict() for interaction in recent_interactions]


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
                
                # Update context processor if available
                load_advanced_features()
                context_processor = initialize_context_processor()
                if ADVANCED_FEATURES_AVAILABLE and context_processor and speaker_index in context_processor.speaker_profiles:
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
    # Use reload only in development, not when launched via script
    reload_mode = os.getenv("MIRA_DEV_MODE", "false").lower() == "true"
    uvicorn.run("mira:app", host="0.0.0.0", port=8000, reload=reload_mode)