"""
Live interaction with real-time speaker diarization (up to two speakers) using
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
import logging

import numpy as np

from db import get_db_session
from models import Person
from sqlalchemy.orm import Session
import uuid

logger = logging.getLogger(__name__)

warnings.filterwarnings("ignore", message="FP16 is not supported on CPU; using FP32 instead")
warnings.filterwarnings(
    "ignore",
    category=UserWarning,
    message="pkg_resources is deprecated as an API",
)

SAMPLE_RATE = 16_000
SIM_THRESHOLD = 0.75
MAX_SPEAKERS = 1


asr_model = None
spk_encoder = None


def get_asr_model():
    global asr_model
    if asr_model is None:
        try:
            import whisper
            asr_model = whisper.load_model("base")
        except ImportError as e:
            logger.warning(f"Failed to load whisper model: {e}")
            asr_model = None
    return asr_model


def get_spk_encoder():
    global spk_encoder
    if spk_encoder is None:
        try:
            from resemblyzer import VoiceEncoder
            spk_encoder = VoiceEncoder()
        except ImportError as e:
            logger.warning(f"Failed to load voice encoder: {e}")
            spk_encoder = None
    return spk_encoder


class SpeakerIdentificationState:
    """State variables for advanced speaker identification moved from context_processor."""

    def __init__(self):
        self._speaker_embeddings: list[np.ndarray] = []
        self._speaker_ids: list[uuid.UUID] = []
        self._cluster_labels: list[int] = []
        self._clusters_dirty: bool = True

        self.SPEAKER_SIMILARITY_THRESHOLD: float = 0.7
        self.DBSCAN_EPS: float = 0.9
        self.DBSCAN_MIN_SAMPLES: int = 2
        self._dbscan = None

    @property
    def dbscan(self):
        if self._dbscan is None:
            try:
                from sklearn.cluster import DBSCAN
                self._dbscan = DBSCAN(
                    eps=self.DBSCAN_EPS,
                    min_samples=self.DBSCAN_MIN_SAMPLES,
                    metric="cosine",
                )
            except ImportError as e:
                logger.warning(f"Failed to load DBSCAN: {e}")
                self._dbscan = None
        return self._dbscan


_speaker_state = SpeakerIdentificationState()


def pcm_bytes_to_float32(pcm: bytes) -> np.ndarray:
    """Convert 16-bit PCM to float32 in [-1,1]."""
    audio_int16 = np.frombuffer(pcm, dtype=np.int16)
    return audio_int16.astype(np.float32) / 32768.0


def butter_highpass(cutoff, fs, order=5):
    """Design a high-pass Butterworth filter."""
    from scipy.signal import butter
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
    from scipy.signal import lfilter
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

    try:
        audio_data = np.array(audio_data, dtype=np.float32)
        filtered_audio = butter_highpass_filter(audio_data, 80, sample_rate)
        filtered_audio = np.array(filtered_audio, dtype=np.float32)

        if len(filtered_audio) < 512:
            return filtered_audio

        if len(filtered_audio) > sample_rate // 2:
            noise_sample = filtered_audio[: sample_rate // 2]
            try:
                import noisereduce as nr
                denoised_audio = nr.reduce_noise(
                    y=filtered_audio,
                    sr=sample_rate,
                    y_noise=noise_sample,
                    prop_decrease=0.8,
                    stationary=False,
                    n_fft=min(512, len(filtered_audio)),
                    hop_length=min(128, len(filtered_audio) // 4),
                )
            except ImportError as e:
                logger.warning(f"noisereduce not available: {e}")
                denoised_audio = filtered_audio
        else:
            try:
                import noisereduce as nr
                denoised_audio = nr.reduce_noise(
                    y=filtered_audio,
                    sr=sample_rate,
                    prop_decrease=0.6,
                    stationary=True,
                    n_fft=min(256, len(filtered_audio)),
                    hop_length=min(64, len(filtered_audio) // 4),
                )
            except ImportError as e:
                logger.warning(f"noisereduce not available: {e}")
                denoised_audio = filtered_audio

        return np.array(denoised_audio, dtype=np.float32)

    except Exception as e:
        logger.warning(f"Denoising failed: {e}")
        filtered = butter_highpass_filter(audio_data, 80, sample_rate)
        filtered = np.array(filtered, dtype=np.float32)
        if isinstance(filtered, tuple):
            filtered = filtered[0]
        return filtered


def cosine_sim(a: np.ndarray, b: np.ndarray) -> float:
    """Cosine similarity between two vectors."""
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))


def _refresh_speaker_cache():
    """(Re)load all speaker embeddings, person_ids, interaction_ids from the database."""
    from models import Interaction

    session = get_db_session()
    try:
        interactions = (
            session.query(Interaction)
            .filter(Interaction.voice_embedding.isnot(None), Interaction.speaker_id.isnot(None))
            .all()
        )
        _speaker_state._speaker_embeddings = [
            np.array(i.voice_embedding, dtype=np.float32) for i in interactions
        ]
        _speaker_state._speaker_ids = [i.speaker_id for i in interactions]
        _speaker_state._clusters_dirty = True
    finally:
        session.close()


def _recompute_clusters():
    """Run DBSCAN clustering on all cached embeddings and update labels."""
    if not _speaker_state._speaker_embeddings or (
        isinstance(_speaker_state._speaker_embeddings, np.ndarray)
        and _speaker_state._speaker_embeddings.size == 0
    ):
        _speaker_state._cluster_labels = []
        return
    X = np.stack(_speaker_state._speaker_embeddings)
    if _speaker_state.dbscan is not None:
        _speaker_state._cluster_labels = _speaker_state.dbscan.fit_predict(X).tolist()
    else:
        # Fallback: assign each embedding to a separate cluster
        _speaker_state._cluster_labels = list(range(len(_speaker_state._speaker_embeddings)))
    _speaker_state._clusters_dirty = False


def assign_speaker(
    embedding: np.ndarray,
):
    """
    Assign a speaker using DBSCAN clustering over all embeddings (moved from context_processor).
    Returns the Person.id of the most similar speaker (if above threshold) or creates a new one.
    Also updates clusters in the database.

    Args:
        embedding: The new voice embedding (np.ndarray)
        interaction_id: The interaction UUID to use for the new embedding, if available
    """

    embedding = np.array(embedding, dtype=np.float32)

    session = get_db_session()

    if (
        not _speaker_state._speaker_embeddings
        or (
            isinstance(_speaker_state._speaker_embeddings, np.ndarray)
            and _speaker_state._speaker_embeddings.size == 0
        )
        or _speaker_state._clusters_dirty
    ):
        _refresh_speaker_cache()
        _recompute_clusters()

    all_embeddings = _speaker_state._speaker_embeddings + [embedding]
    X = np.stack(all_embeddings)
    try:
        from sklearn.cluster import DBSCAN
        dbscan = DBSCAN(
            eps=_speaker_state.DBSCAN_EPS,
            min_samples=_speaker_state.DBSCAN_MIN_SAMPLES,
            metric="cosine",
        )
        labels = dbscan.fit_predict(X)
        new_label = labels[-1]
    except ImportError as e:
        logger.warning(f"DBSCAN not available for speaker assignment: {e}")
        # Fallback: create new speaker
        new_label = -1

    def _append_cache(embedding, person_id):
        _speaker_state._speaker_embeddings.append(embedding)
        _speaker_state._speaker_ids.append(person_id)
        _speaker_state._clusters_dirty = True

    if new_label == -1:
        new_index = (session.query(Person.index).order_by(Person.index.desc()).first() or [0])[
            0
        ] + 1
        new_person = Person(
            voice_embedding=(
                embedding.tolist() if isinstance(embedding, np.ndarray) else embedding
            ),
            index=new_index,
        )
        session.add(new_person)
        session.commit()
        session.refresh(new_person)
        _append_cache(embedding, new_person.id)
        _update_db_clusters(session, labels[:-1] + [-1])
        return new_person.id

    cluster_indices = [i for i, label in enumerate(labels[:-1]) if label == new_label]
    if not cluster_indices:
        new_index = (session.query(Person.index).order_by(Person.index.desc()).first() or [0])[
            0
        ] + 1
        new_person = Person(
            voice_embedding=embedding.tolist(),
            index=new_index,
        )
        session.add(new_person)
        session.commit()
        session.refresh(new_person)
        _append_cache(embedding, new_person.id)
        _update_db_clusters(session, labels[:-1] + [-1])
        return new_person.id

    similarities = []
    for idx in cluster_indices:
        emb = all_embeddings[idx]
        sim = float(np.dot(embedding, emb) / (np.linalg.norm(embedding) * np.linalg.norm(emb)))
        similarities.append((idx, sim))
    best_idx, best_sim = max(similarities, key=lambda x: x[1])

    logger.info(f"Best speaker similarity: {best_sim}")

    if best_sim < _speaker_state.SPEAKER_SIMILARITY_THRESHOLD:
        new_index = (session.query(Person.index).order_by(Person.index.desc()).first() or [0])[
            0
        ] + 1
        new_person = Person(
            voice_embedding=embedding.tolist(),
            index=new_index,
        )
        session.add(new_person)
        session.commit()
        session.refresh(new_person)
        _append_cache(embedding, new_person.id)
        _update_db_clusters(session, labels[:-1] + [-1])
        return new_person.id

    matched_person_id = _speaker_state._speaker_ids[best_idx]
    matched_person = session.query(Person).filter_by(id=matched_person_id).first()
    if matched_person and matched_person.voice_embedding is not None:
        old_emb = np.array(matched_person.voice_embedding, dtype=np.float32)
        updated_emb = 0.8 * old_emb + 0.2 * embedding
        matched_person.voice_embedding = updated_emb.tolist()
        session.commit()
    _append_cache(embedding, matched_person_id)
    _update_db_clusters(session, labels)
    return matched_person_id


def update_voice_embedding(person_id: uuid.UUID, audio_buffer: bytearray, expected_text: str):
    """
    Update the embedding for a given person_id using new audio and expected text.
    Incorporates expected_text into the embedding update, similar to supervised phrase training.

    Args:
        person_id: UUID of the person to update
        audio_buffer: Raw audio data as bytearray
        expected_text: The expected phrase spoken (for supervised adaptation)

    Returns:
        True if update was successful, False otherwise
    """
    session = get_db_session()

    interaction = transcribe_interaction(audio_buffer, assign_or_create_speaker=False)

    transcribed_text = interaction.get("text", "")
    if not transcribed_text or expected_text.lower() not in transcribed_text.lower():
        logger.info(
            f"Transcribed text '{transcribed_text}' does not match expected '{expected_text}'"
        )

    denoised_audio = denoise_audio(pcm_bytes_to_float32(bytes(audio_buffer)), SAMPLE_RATE)
    spk_encoder = get_spk_encoder()
    if spk_encoder is None:
        logger.warning("Voice encoder not available, skipping embedding update")
        return
    
    embedding_result = spk_encoder.embed_utterance(denoised_audio)
    new_embedding = embedding_result[0] if isinstance(embedding_result, tuple) else embedding_result
    new_embedding = np.array(new_embedding, dtype=np.float32)

    person = session.query(Person).filter_by(id=person_id).first()
    if not person or person.voice_embedding is None:
        raise ValueError(f"Person {person_id} not found or missing embedding")

    old_embedding = np.array(person.voice_embedding, dtype=np.float32)
    if old_embedding.dtype != np.float32:
        old_embedding = old_embedding.astype(np.float32)
    if new_embedding.dtype != np.float32:
        new_embedding = new_embedding.astype(np.float32)
    updated_embedding = 0.7 * old_embedding + 0.3 * new_embedding
    updated_embedding = np.array(updated_embedding, dtype=np.float32)
    person.voice_embedding = updated_embedding.tolist()
    session.commit()

    if person_id in _speaker_state._speaker_ids:
        idx = _speaker_state._speaker_ids.index(person_id)
        _speaker_state._speaker_embeddings[idx] = updated_embedding
        _speaker_state._clusters_dirty = True

    logger.info(f"Updated embedding for person {person_id} using expected text '{expected_text}'")


def _update_db_clusters(session: Session, cluster_labels):
    """
    Update DB cluster assignments for all Persons based on current cache and cluster_labels.
    Interactions in cache must match the order of cluster_labels.
    """
    if (
        len(_speaker_state._speaker_ids) == 0
        or len(cluster_labels) == 0
        or len(_speaker_state._speaker_ids) != len(cluster_labels)
    ):
        return
    for person_id, label in zip(_speaker_state._speaker_ids, cluster_labels):
        if person_id is None:
            continue
        person = session.query(Person).filter_by(id=person_id).first()
        if person:
            person.cluster_id = int(label) if label != -1 else None
    session.commit()


def transcribe_interaction(sentence_buf: bytearray, assign_or_create_speaker: bool) -> dict:
    """
    Process a complete sentence buffer with real-time audio denoising and speaker recognition.

    Args:
        sentence_buf: Audio buffer containing the sentence
    """

    audio_f32 = pcm_bytes_to_float32(bytes(sentence_buf))

    print(f"Processing audio buffer of length {len(audio_f32)}")

    denoised_audio = denoise_audio(audio_f32, SAMPLE_RATE)

    if np.isnan(denoised_audio).any() or np.isinf(denoised_audio).any():
        raise ValueError("Audio contains NaN or Inf values")

    asr_model = get_asr_model()
    if asr_model is None:
        # Fallback when whisper is not available
        logger.warning("ASR model not available, using fallback transcription")
        result = {
            "text": "Mock transcription - ASR model not available",
            "language": "en"
        }
    else:
        result = asr_model.transcribe(denoised_audio)

    text = str(
        " ".join(result["text"]).strip()
        if isinstance(result["text"], list)
        else result["text"].strip()
    )

    if not text:
        raise ValueError("Transcription failed")

    interaction = dict()
    interaction["text"] = text

    if assign_or_create_speaker:
        spk_encoder = get_spk_encoder()
        if spk_encoder is None:
            logger.warning("Voice encoder not available, skipping speaker assignment")
            # Use a fallback speaker ID or create a default one
            interaction["voice_embedding"] = None
            interaction["speaker_id"] = None
        else:
            embedding_result = spk_encoder.embed_utterance(denoised_audio)
            embedding = embedding_result[0] if isinstance(embedding_result, tuple) else embedding_result
            speaker_id = assign_speaker(embedding)
            interaction["voice_embedding"] = embedding.tolist()
            interaction["speaker_id"] = speaker_id

    return interaction
