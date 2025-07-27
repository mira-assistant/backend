import uuid
from fastapi import FastAPI, HTTPException
from run_inference import send_prompt
from models import Interaction, Person
from db import get_db_session
import uvicorn
from sentence_processor import transcribe_interaction
from enhanced_context_processor import create_enhanced_context_processor, process_interaction_enhanced
from context_config import DEFAULT_CONFIG
import numpy as np
from datetime import datetime
import logging
import json


# Initialize enhanced context processor
context_processor = create_enhanced_context_processor(DEFAULT_CONFIG)

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

if __name__ == "__main__":
    uvicorn.run("mira:app", host="0.0.0.0", port=8000, reload=True)

app = FastAPI()

status: dict = {
    "version": "2.0.0",  # Updated version for enhanced features
    "listening_clients": list(),
    "enabled": False,
    "features": {
        "advanced_nlp": True,
        "speaker_clustering": True,
        "context_summarization": True,
        "database_integration": True
    }
}


@app.get("/")
def root():
    return status


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
def process_interaction(sentence_buf: bytes):  # Change from bytearray to bytes
    """Process interaction - transcribe sentence and identify speaker only."""
    try:
        # Convert bytes to bytearray for processing
        sentence_buf_array = bytearray(sentence_buf)
        
        # Transcribe with voice embedding
        transcription_result = transcribe_interaction(sentence_buf_array)
        
        # Create database interaction
        interaction = Interaction(
            user_id=transcription_result["speaker"],  # Use user_id for backward compatibility
            text=transcription_result["text"]
        )

        # Check for shutdown command first (moved here for immediate response)
        if "mira" in interaction.text.lower() and any(
            cancelCMD in interaction.text.lower() for cancelCMD in ("cancel", "exit")
        ):
            disable_service()
            return status

        db = get_db_session()
        db.add(interaction)
        db.commit()
        db.refresh(interaction)
        db.close()

        # Return basic transcription result
        response = {
            "id": str(interaction.id),
            "speaker": interaction.speaker,  # This uses the property
            "text": interaction.text,
            "timestamp": interaction.timestamp.isoformat(),
        }
        
        return response
        
    except Exception as e:
        logger.error(f"Error processing interaction: {e}")
        raise HTTPException(status_code=500, detail=f"Processing failed: {str(e)}")


@app.post("/inference")
def inference_endpoint(interaction_id):
    """Enhanced inference endpoint with context integration."""
    try:
        interaction = (
            get_db_session().query(Interaction).filter_by(id=uuid.UUID(interaction_id)).first()
        )

        if not interaction:
            raise HTTPException(status_code=404, detail="Interaction not found")

        # Extract voice embedding if available from transcription
        voice_embedding = None
        # Note: voice_embedding would come from the transcription step
        
        # Use enhanced context processing with the new function
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
            "entities": getattr(context_processor.interaction_history[-1], 'entities', None) if context_processor.interaction_history else None,
            "sentiment": getattr(context_processor.interaction_history[-1], 'sentiment', None) if context_processor.interaction_history else None
        })
        
        return response
        
    except Exception as e:
        logger.error(f"Error in inference: {e}")
        raise HTTPException(status_code=500, detail=f"Inference failed: {str(e)}")


@app.get("/context/speakers")
def get_speaker_summary():
    """Get summary of all tracked speakers."""
    return context_processor.get_speaker_summary()


@app.get("/context/history")
def get_interaction_history(limit: int = 10):
    """Get recent interaction history."""
    recent_interactions = context_processor.interaction_history[-limit:]
    return [interaction.to_dict() for interaction in recent_interactions]


@app.post("/context/identify_speaker")
def identify_speaker(speaker_index: int, name: str):
    """Manually identify a speaker."""
    try:
        db = get_db_session()
        person = db.query(Person).filter_by(speaker_index=speaker_index).first()
        
        if person:
            person.name = name
            person.is_identified = True
            db.commit()
            
            # Update context processor
            if speaker_index in context_processor.speaker_profiles:
                context_processor.speaker_profiles[speaker_index].name = name
                context_processor.speaker_profiles[speaker_index].is_identified = True
                
            db.close()
            return {"message": f"Speaker {speaker_index} identified as {name}"}
        else:
            raise HTTPException(status_code=404, detail="Speaker not found")
            
    except Exception as e:
        logger.error(f"Error identifying speaker: {e}")
        raise HTTPException(status_code=500, detail=f"Speaker identification failed: {str(e)}")