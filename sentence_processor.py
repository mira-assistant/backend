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

import numpy as np
import whisper
from resemblyzer import VoiceEncoder
import noisereduce as nr
from scipy.signal import butter, lfilter

from db import get_db_session
from models import Person

warnings.filterwarnings("ignore", message="FP16 is not supported on CPU; using FP32 instead")
warnings.filterwarnings(
    "ignore",
    category=UserWarning,
    message="pkg_resources is deprecated as an API",
)

# ---------- Constants ----------
SAMPLE_RATE = 16_000
SIM_THRESHOLD = 0.75
MAX_SPEAKERS = 1

# ---------- Global model instances (loaded once) ----------
_asr_model = None
_spk_encoder = None
_speaker_centroids: list[np.ndarray] = []


def get_models():
    """Get or initialize the ASR model and speaker encoder (singleton pattern)"""
    global _asr_model, _spk_encoder

    if _asr_model is None:
        _asr_model = whisper.load_model("base")

    if _spk_encoder is None:
        _spk_encoder = VoiceEncoder()

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


def assign_speaker(new_embedding: np.ndarray):
    """Assign embedding to a speaker index; update centroids online."""

    db = get_db_session()

    users = db.query(Person).all()

    for user in users:
        voice_embedding = np.array(getattr(user, "voice_embedding"), dtype=np.float32)

        similarity_score = cosine_sim(voice_embedding, new_embedding)

        if similarity_score >= SIM_THRESHOLD:
            # Weighted update: new_embedding = alpha * new_embedding + (1 - alpha) * old_embedding
            # alpha decreases as number of interactions increases (e.g., alpha = 1 / (n + 1))
            num_interactions = len(user.interactions)
            alpha = 1.0 / (num_interactions + 1)
            updated_embedding = alpha * new_embedding + (1 - alpha) * voice_embedding
            user.voice_embedding = updated_embedding.tolist()
            db.commit()

            return user.id

    # If no existing speaker matches, create a new one
    new_user = Person(voice_embedding=new_embedding.tolist(), speaker_index=len(users))
    db.add(new_user)
    db.commit()
    db.refresh(new_user)
    return new_user.id


def transcribe_interaction(sentence_buf: bytearray) -> dict | None:
    """
    Process a complete sentence buffer with real-time audio denoising and speaker recognition.
    """
    # Use cached models instead of loading them each time
    asr_model, spk_encoder = get_models()

    interaction = dict()

    audio_f32 = pcm_bytes_to_float32(bytes(sentence_buf))
    if len(audio_f32) < SAMPLE_RATE * 1:
        # Not enough float32 samples (1 second worth)
        return None

    # Apply real-time audio denoising to filter out white noise
    denoised_audio = denoise_audio(audio_f32, SAMPLE_RATE)
    denoised_audio = denoised_audio.astype(np.float32)

    if np.isnan(denoised_audio).any() or np.isinf(denoised_audio).any():
        raise ValueError("Audio contains NaN or Inf values")

    result = asr_model.transcribe(denoised_audio)
    text = str(
        " ".join(result["text"]).strip()
        if isinstance(result["text"], list)
        else result["text"].strip()
    )

    if not text:
        return None

    embedding_result = spk_encoder.embed_utterance(denoised_audio)
    embedding = embedding_result[0] if isinstance(embedding_result, tuple) else embedding_result
    # speaker_id = assign_speaker(embedding)

    interaction["text"] = text
    interaction["voice_embedding"] = embedding.tolist()

    return interaction
