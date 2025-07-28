"""
Live transcription with real-time speaker diarization (up to two speakers) using
local Whisper, WebRTC-VAD, and Resemblyzer with real-time audio denoising.

Supports multilingual speech recognition including English and Indian languages.
Code-mixed speech (e.g., English + Hindi/Tamil/etc.) is automatically detected.
Real-time audio denoising filters out white noise and background sounds.

How it works
------------
• Captures 30 ms frames from the microphone.
• WebRTC-VAD detects speech; ≥ 2 s of silence marks sentence boundary.
• Real-time audio denoising removes white noise and background sounds from audio.
• Each sentence is transcribed by Whisper locally (no API) with automatic language detection.
• Indian names and words are recognized alongside English through multilingual model.
• The sentence audio is embedded with Resemblyzer.
• A lightweight online clustering assignment labels sentences
  as Speaker 1 or Speaker 2 by cosine similarity to running centroids.
"""

from __future__ import annotations

import warnings
from typing import List

import numpy as np
import whisper
from resemblyzer import VoiceEncoder
import noisereduce as nr
from scipy.signal import butter, lfilter

warnings.filterwarnings(
    "ignore", message="FP16 is not supported on CPU; using FP32 instead"
)
warnings.filterwarnings(
    "ignore", category=UserWarning, message="pkg_resources is deprecated as an API"
)

# ---------- Constants ----------
SAMPLE_RATE = 16_000
SIM_THRESHOLD = 0.75
MAX_SPEAKERS = 1

# ---------- Global model instances (loaded once) ----------
_asr_model = None
_spk_encoder = None
_speaker_centroids = []

def get_models():
    """Get or initialize the ASR model and speaker encoder (singleton pattern)"""
    global _asr_model, _spk_encoder
    
    if _asr_model is None:
        print("Loading Whisper ASR model...")
        _asr_model = whisper.load_model("base")
        print("✅ Whisper model loaded")
    
    if _spk_encoder is None:
        print("Loading voice encoder model...")
        _spk_encoder = VoiceEncoder()
        print("✅ Voice encoder model loaded")
    
    return _asr_model, _spk_encoder


def pcm_bytes_to_float32(pcm: bytes) -> np.ndarray:
    """Convert 16-bit PCM to float32 in [-1,1]."""
    audio_int16 = np.frombuffer(pcm, dtype=np.int16)
    return audio_int16.astype(np.float32) / 32768.0


def butter_highpass(cutoff, fs, order=5):
    """Design a high-pass Butterworth filter."""
    nyquist = 0.5 * fs
    if cutoff <= 0 or cutoff >= nyquist:
        raise ValueError(
            f"Cutoff frequency must be between 0 and Nyquist ({nyquist} Hz), got {cutoff}"
        )
    normal_cutoff = cutoff / nyquist
    result = butter(order, normal_cutoff, btype="high", analog=False)
    if result is None:
        raise ValueError("Butterworth filter design failed; check cutoff frequency.")
    b, a = result[:2]
    return b, a


def butter_highpass_filter(data, cutoff, fs, order=5):
    """Apply a high-pass Butterworth filter to the data."""
    b, a = butter_highpass(cutoff, fs, order=order)
    y = lfilter(b, a, data)
    return y


def denoise_audio(audio_data: np.ndarray, sample_rate: int = SAMPLE_RATE) -> np.ndarray:
    """
    Apply real-time audio denoising to remove white noise and background sounds.

    Args:
        audio_data: Input audio signal as float32 array
        sample_rate: Sample rate of the audio

    Returns:
        Denoised audio signal
    """
    # Apply noise reduction using noisereduce library
    try:
        # Apply high-pass filter to remove low-frequency noise (e.g., 80 Hz cutoff)
        filtered_audio = butter_highpass_filter(audio_data, 80, sample_rate)

        # Apply noise reduction - reduce stationary noise
        # Use the first 0.5 seconds as noise sample for adaptive filtering
        if len(filtered_audio) > sample_rate // 2:
            noise_sample = filtered_audio[: sample_rate // 2]
            denoised_audio = nr.reduce_noise(
                y=filtered_audio,
                sr=sample_rate,
                y_noise=noise_sample,
                prop_decrease=0.8,  # Reduce noise by 80%
                stationary=False,  # Non-stationary noise reduction
            )
        else:
            # For very short audio, just apply basic noise reduction
            denoised_audio = nr.reduce_noise(
                y=filtered_audio,
                sr=sample_rate,
                prop_decrease=0.6,  # Reduce noise by 60%
                stationary=True,  # Stationary noise reduction for short clips
            )

        return denoised_audio

    except Exception as e:
        # If denoising fails, return the original audio with just high-pass filtering
        print(f"Denoising warning: {e}")
        filtered = butter_highpass_filter(audio_data, 80, sample_rate)
        if isinstance(filtered, tuple):
            filtered = filtered[0]
        return filtered


def cosine_sim(a: np.ndarray, b: np.ndarray) -> float:
    """Cosine similarity between two vectors."""
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))


def assign_speaker(embedding: np.ndarray, centroids: List[np.ndarray] = None) -> int:
    """Assign embedding to a speaker index; update centroids online."""
    global _speaker_centroids
    
    # Use global centroids if none provided
    if centroids is None:
        centroids = _speaker_centroids
    
    if not centroids:
        centroids.append(embedding.copy())
        return 0

    sims = [cosine_sim(embedding, c) for c in centroids]
    best_idx = int(np.argmax(sims))
    best_sim = sims[best_idx]

    # New speaker if similarity is low and we haven't reached MAX_SPEAKERS
    if best_sim < SIM_THRESHOLD and len(centroids) < MAX_SPEAKERS:
        centroids.append(embedding.copy())
        return len(centroids) - 1

    # Update centroid (simple running average)
    centroids[best_idx] = (centroids[best_idx] + embedding) / 2.0
    return best_idx


def transcribe_interaction(sentence_buf: bytearray) -> dict:
    """
    Process a complete sentence buffer with real-time audio denoising and enhanced speaker recognition.
    """
    # Use cached models instead of loading them each time
    asr_model, spk_encoder = get_models()

    interaction = dict()

    audio_f32 = pcm_bytes_to_float32(bytes(sentence_buf))
    if len(audio_f32) < SAMPLE_RATE * 1.5:
        # Not enough float32 samples (1.5 seconds worth)
        raise ValueError(
            "Audio buffer too short for processing; must be at least 1.5 seconds."
        )

    # Apply real-time audio denoising to filter out white noise
    denoised_audio = denoise_audio(audio_f32, SAMPLE_RATE)
    denoised_audio = denoised_audio.astype(np.float32)

    result = asr_model.transcribe(denoised_audio)
    text = (
        " ".join(result["text"]).strip()
        if isinstance(result["text"], list)
        else result["text"].strip()
    )

    if text:
        embedding_result = spk_encoder.embed_utterance(denoised_audio)
        embedding = (
            embedding_result[0]
            if isinstance(embedding_result, tuple)
            else embedding_result
        )
        spk_idx = assign_speaker(embedding)

        interaction["speaker"] = spk_idx + 1
        interaction["text"] = text
        interaction["voice_embedding"] = (
            embedding.tolist() if hasattr(embedding, "tolist") else embedding
        )

        return interaction
    else:
        raise ValueError("No text transcribed from audio buffer; check audio quality.")
