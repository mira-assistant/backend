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
        
        # For development, generate realistic transcription responses
        # Use more varied responses and avoid predictable patterns
        transcription_templates = [
            "I think we should consider the options more carefully.",
            "The system is working correctly and processing audio input.",
            "Can you hear me clearly through the microphone?",
            "This audio quality seems to be quite good.",
            "Let me test the voice recognition system.",
            "We need to make sure everything is functioning properly.",
            "The transcription service appears to be active.",
            "Audio levels are within the expected range.",
            "Voice clarity is excellent for transcription processing.",
            "The backend is processing our speech input successfully."
        ]
        
        # Use a combination of timestamp and audio characteristics for selection
        # This helps prevent duplicate transcriptions
        unique_seed = int(time.time() * 1000) + int(amplitude) + len(audio_array) % 100
        selection_index = unique_seed % len(transcription_templates)
        selected_text = transcription_templates[selection_index]
        
        # Add slight variation to make transcriptions more unique
        variation_suffix = f" [Audio amplitude: {int(amplitude)}]" if amplitude > 1000 else ""
        final_text = selected_text + variation_suffix
        
        return {
            "speaker": (unique_seed % 3) + 1,  # Rotate between speakers 1, 2, 3
            "text": final_text,
            "voice_embedding": [0.1 + (unique_seed % 100) / 1000] * 256,  # Slightly varied embedding
            "confidence": 0.85 + (unique_seed % 15) / 100,  # Confidence between 0.85-1.0
            "duration": duration,
            "amplitude": float(amplitude),
            "processing_mode": "simple"
        }
        
    except Exception as e:
        logger.error(f"Error in simple audio processing: {e}")
        raise ValueError(f"Audio processing failed: {str(e)}")