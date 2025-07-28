"""
Simple audio processor for testing and development.
This provides basic functionality without heavy AI dependencies.
"""

import numpy as np
import time
from typing import Dict, Any
import logging

logger = logging.getLogger(__name__)

def process_audio_simple(audio_bytes: bytes) -> Dict[str, Any]:
    """
    Simple audio processor that generates mock transcriptions for testing.
    This is a fallback when the heavy AI models can't be loaded.
    """
    try:
        # Convert audio bytes to numpy array for basic processing
        audio_array = np.frombuffer(audio_bytes, dtype=np.int16)
        
        # Basic audio analysis - just check if there's actually audio data
        if len(audio_array) == 0:
            raise ValueError("No audio data received")
        
        # Check for silence (very low amplitude)
        amplitude = np.mean(np.abs(audio_array))
        if amplitude < 100:  # Very quiet audio
            logger.warning("Audio amplitude is very low, may be silence")
        
        # Generate a simple response based on audio characteristics
        duration = len(audio_array) / 16000.0  # Assuming 16kHz sample rate
        
        # For development, generate a realistic transcription response
        transcriptions = [
            "Hello, this is a test of the voice transcription system.",
            "The microphone is picking up audio correctly.",
            "Voice recognition is working as expected.",
            "This is speaker recognition test number {}".format(int(time.time()) % 10),
            "Audio processing is functioning normally.",
        ]
        
        # Use audio characteristics to select a transcription
        selection_index = int(amplitude) % len(transcriptions)
        selected_text = transcriptions[selection_index]
        
        return {
            "speaker": 1,  # Default to speaker 1
            "text": selected_text,
            "voice_embedding": [0.1] * 256,  # Mock embedding
            "confidence": 0.95,
            "duration": duration,
            "amplitude": float(amplitude)
        }
        
    except Exception as e:
        logger.error(f"Error in simple audio processing: {e}")
        raise ValueError(f"Audio processing failed: {str(e)}")