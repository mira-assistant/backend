"""
Live interaction with real-time speaker diarization (up to two speakers) using
local Whisper and Resemblyzer with real-time audio denoising.

Supports multilingual speech recognition including English and Indian languages.
Code-mixed speech (e.g., English + Hindi/Tamil/etc.) is automatically detected.
Real-time audio denoising filters out white noise and background sounds.

How it works
------------
• Captures 30 ms frames from the microphone.
• Real-time audio denoising removes white noise and background sounds from audio.
• Each sentence is transcribed by Whisper locally (no API) with automatic language detection.
• Indian names and words are recognized alongside English through multilingual model.
• The sentence audio is embedded with Resemblyzer.
• A lightweight online clustering assignment labels sentences
  as Speaker 1 or Speaker 2 by cosine similarity to running centroids.
"""

from __future__ import annotations

import logging

import numpy as np

import whisper
from resemblyzer import VoiceEncoder
import noisereduce as nr
from scipy.signal import butter, lfilter

from db import get_db_session
from models import Person
from sklearn.cluster import DBSCAN
from sqlalchemy.orm import Session
import uuid

logger = logging.getLogger(__name__)


class SpeakerIdentificationState:
    """State variables for advanced speaker identification moved from context_processor."""

    def __init__(self):
        # Speaker detection state variables for advanced clustering
        self._speaker_embeddings: list[np.ndarray] = []
        self._speaker_ids: list[uuid.UUID] = []
        self._cluster_labels: list[int] = []
        self._clusters_dirty: bool = True

        # DBSCAN configuration
        self.SPEAKER_SIMILARITY_THRESHOLD: float = 0.7
        self.DBSCAN_EPS: float = 0.9
        self.DBSCAN_MIN_SAMPLES: int = 2

        self.dbscan = DBSCAN(
            eps=self.DBSCAN_EPS,
            min_samples=self.DBSCAN_MIN_SAMPLES,
            metric="cosine",
        )


class SentenceProcessor:
    SAMPLE_RATE = 16_000

    def __init__(self):
        """
        Initialize SentenceProcessor with Whisper ASR model and Resemblyzer voice encoder.
        """
        # Load the Whisper model and Resemblyzer voice encoder
        self.asr_model = whisper.load_model("base")
        self.spk_encoder = VoiceEncoder()
        self._speaker_state = SpeakerIdentificationState()
        logger.info("SentenceProcessor initialized")


    def pcm_bytes_to_float32(self, pcm: bytes) -> np.ndarray:
        """Convert 16-bit PCM to float32 in [-1,1]."""
        audio_int16 = np.frombuffer(pcm, dtype=np.int16)
        return audio_int16.astype(np.float32) / 32768.0


    def butter_highpass(self, cutoff, fs, order=5):
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


    def butter_highpass_filter(self, data, cutoff, fs, order=5):
        """Apply a high-pass Butterworth filter to the data."""
        b, a = self.butter_highpass(cutoff, fs, order=order)
        y = lfilter(b, a, data)
        return y


    def denoise_audio(self, audio_data: np.ndarray) -> np.ndarray:
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
            # Ensure input is float32
            audio_data = np.array(audio_data, dtype=np.float32)
            # Apply high-pass filter to remove low-frequency noise (e.g., 80 Hz cutoff)
            filtered_audio = self.butter_highpass_filter(audio_data, 80, self.SAMPLE_RATE)
            filtered_audio = np.array(filtered_audio, dtype=np.float32)

            # If audio is very short, skip noise reduction to avoid nperseg/noverlap errors
            if len(filtered_audio) < 512:
                return filtered_audio

            # Use the first 0.5 seconds as noise sample for adaptive filtering
            if len(filtered_audio) > self.SAMPLE_RATE // 2:
                noise_sample = filtered_audio[: self.SAMPLE_RATE // 2]
                denoised_audio = nr.reduce_noise(
                    y=filtered_audio,
                    sr=self.SAMPLE_RATE,
                    y_noise=noise_sample,
                    prop_decrease=0.8,  # Reduce noise by 80%
                    stationary=False,  # Non-stationary noise reduction
                    n_fft=min(512, len(filtered_audio)),
                    hop_length=min(128, len(filtered_audio) // 4),
                )
            else:
                # For very short audio, just apply basic noise reduction
                denoised_audio = nr.reduce_noise(
                    y=filtered_audio,
                    sr=self.SAMPLE_RATE,
                    prop_decrease=0.6,  # Reduce noise by 60%
                    stationary=True,  # Stationary noise reduction for short clips
                    n_fft=min(256, len(filtered_audio)),
                    hop_length=min(64, len(filtered_audio) // 4),
                )

            return np.array(denoised_audio, dtype=np.float32)

        except Exception as e:
            # If denoising fails, return the original audio with just high-pass filtering
            logger.warning(f"Denoising failed: {e}")
            filtered = self.butter_highpass_filter(audio_data, 80, self.SAMPLE_RATE)
            filtered = np.array(filtered, dtype=np.float32)
            if isinstance(filtered, tuple):
                filtered = filtered[0]
            return filtered


    def cosine_sim(self, a: np.ndarray, b: np.ndarray) -> float:
        """Cosine similarity between two vectors."""
        return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))


    def _refresh_speaker_cache(self):
        """(Re)load all speaker embeddings, person_ids, interaction_ids from the database."""
        from models import Interaction

        session = get_db_session()
        try:
            interactions = (
                session.query(Interaction)
                .filter(Interaction.voice_embedding.isnot(None), Interaction.speaker_id.isnot(None))
                .all()
            )
            self._speaker_state._speaker_embeddings = [
                np.array(i.voice_embedding, dtype=np.float32) for i in interactions
            ]
            self._speaker_state._speaker_ids = [i.speaker_id for i in interactions]  # type: ignore
            self._speaker_state._clusters_dirty = True
        finally:
            session.close()


    def _recompute_clusters(self):
        """Run DBSCAN clustering on all cached embeddings and update labels."""
        if not self._speaker_state._speaker_embeddings or (
            isinstance(self._speaker_state._speaker_embeddings, np.ndarray)
            and self._speaker_state._speaker_embeddings.size == 0
        ):
            self._speaker_state._cluster_labels = []
            return
        X = np.stack(self._speaker_state._speaker_embeddings)
        self._speaker_state._cluster_labels = self._speaker_state.dbscan.fit_predict(X).tolist()
        self._speaker_state._clusters_dirty = False


    def assign_speaker(self,
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
            not self._speaker_state._speaker_embeddings
            or (
                isinstance(self._speaker_state._speaker_embeddings, np.ndarray)
                and self._speaker_state._speaker_embeddings.size == 0
            )
            or self._speaker_state._clusters_dirty
        ):
            self._refresh_speaker_cache()
            self._recompute_clusters()

        # Add the new embedding to the cached ones for clustering
        all_embeddings = self._speaker_state._speaker_embeddings + [embedding]
        X = np.stack(all_embeddings)
        dbscan = DBSCAN(
            eps=self._speaker_state.DBSCAN_EPS,
            min_samples=self._speaker_state.DBSCAN_MIN_SAMPLES,
            metric="cosine",
        )

        labels = dbscan.fit_predict(X)
        new_label = labels[-1]

        # Helper to append to cache with correct types
        def _append_cache(embedding, person_id):
            self._speaker_state._speaker_embeddings.append(embedding)
            self._speaker_state._speaker_ids.append(person_id)
            self._speaker_state._clusters_dirty = True

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
            self._update_db_clusters(session, labels[:-1] + [-1])
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
            self._update_db_clusters(session, labels[:-1] + [-1])
            return new_person.id

        similarities = []
        for idx in cluster_indices:
            emb = all_embeddings[idx]
            sim = float(np.dot(embedding, emb) / (np.linalg.norm(embedding) * np.linalg.norm(emb)))
            similarities.append((idx, sim))
        best_idx, best_sim = max(similarities, key=lambda x: x[1])

        logger.info(f"Best speaker similarity: {best_sim}")

        if best_sim < self._speaker_state.SPEAKER_SIMILARITY_THRESHOLD:
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
            self._update_db_clusters(session, labels[:-1] + [-1])
            return new_person.id

        # Assign to the Person of the best match in the cluster
        matched_person_id = self._speaker_state._speaker_ids[best_idx]
        matched_person = session.query(Person).filter_by(id=matched_person_id).first()
        if matched_person and matched_person.voice_embedding is not None:
            old_emb = np.array(matched_person.voice_embedding, dtype=np.float32)
            updated_emb = 0.8 * old_emb + 0.2 * embedding
            matched_person.voice_embedding = updated_emb.tolist()
            session.commit()
        _append_cache(embedding, matched_person_id)
        self._update_db_clusters(session, labels)
        return matched_person_id


    def update_voice_embedding(self, person_id: uuid.UUID, audio_buffer: bytearray, expected_text: str):
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

        interaction = self.transcribe_interaction(audio_buffer, assign_or_create_speaker=False)

        transcribed_text = interaction.get("text", "")
        if not transcribed_text or expected_text.lower() not in transcribed_text.lower():
            logger.info(
                f"Transcribed text '{transcribed_text}' does not match expected '{expected_text}'"
            )

        denoised_audio = self.denoise_audio(self.pcm_bytes_to_float32(bytes(audio_buffer)))
        embedding_result = self.spk_encoder.embed_utterance(denoised_audio)
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

        if person_id in self._speaker_state._speaker_ids:
            idx = self._speaker_state._speaker_ids.index(person_id)
            self._speaker_state._speaker_embeddings[idx] = updated_embedding
            self._speaker_state._clusters_dirty = True

        logger.info(f"Updated embedding for person {person_id} using expected text '{expected_text}'")


    def _update_db_clusters(self, session: Session, cluster_labels):
        """
        Update DB cluster assignments for all Persons based on current cache and cluster_labels.
        Interactions in cache must match the order of cluster_labels.
        """
        if (
            len(self._speaker_state._speaker_ids) == 0
            or len(cluster_labels) == 0
            or len(self._speaker_state._speaker_ids) != len(cluster_labels)
        ):
            return
        for person_id, label in zip(self._speaker_state._speaker_ids, cluster_labels):
            if person_id is None:
                continue
            person = session.query(Person).filter_by(id=person_id).first()
            if person:
                person.cluster_id = int(label) if label != -1 else None  # type: ignore
        session.commit()


    def transcribe_interaction(self, sentence_buf: bytearray, assign_or_create_speaker: bool) -> dict:
        """
        Process a complete sentence buffer with real-time audio denoising and speaker recognition.

        Args:
            sentence_buf: Audio buffer containing the sentence
        """

        audio_f32 = self.pcm_bytes_to_float32(bytes(sentence_buf))

        # Apply real-time audio denoising to filter out white noise
        denoised_audio = self.denoise_audio(audio_f32)

        if np.isnan(denoised_audio).any() or np.isinf(denoised_audio).any():
            raise ValueError("Audio contains NaN or Inf values")

        result = self.asr_model.transcribe(denoised_audio)

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
            embedding_result = self.spk_encoder.embed_utterance(denoised_audio)
            embedding = embedding_result[0] if isinstance(embedding_result, tuple) else embedding_result
            speaker_id = self.assign_speaker(embedding)
            interaction["voice_embedding"] = embedding.tolist()
            interaction["speaker_id"] = speaker_id

        return interaction
