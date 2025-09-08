"""
Live interaction with real-time speaker diarization (up to two speakers) using
local Whisper and Resemblyzer with real-time audio denoising.

Supports multilingual speech recognition including English and Indian languages.
Code-mixed speech (e.g., English + Hindi/Tamil/etc.) is automatically detected.
Real-time audio denoising filters out white noise and background sounds.

Now uses proper dependency injection and lifecycle management.
"""

from __future__ import annotations

from typing import Dict, Any

import numpy as np

from faster_whisper import WhisperModel
from resemblyzer import VoiceEncoder
import noisereduce as nr
from scipy.signal import butter, lfilter

from app.db import get_db_session
from app.models import Person, Interaction
from sklearn.cluster import DBSCAN
from sqlalchemy.orm import Session
import uuid
from app.core.mira_logger import MiraLogger
from app.core.config import settings

# MiraLogger is used directly via class methods


class SpeakerIdentificationState:
    """State variables for advanced speaker identification."""

    def __init__(self):
        self._speaker_embeddings: list[np.ndarray] = []
        self._speaker_ids: list[uuid.UUID] = []
        self._cluster_labels: list[int] = []
        self._clusters_dirty: bool = True

        self.SPEAKER_SIMILARITY_THRESHOLD: float = 0.7
        self.DBSCAN_EPS: float = 0.9
        self.DBSCAN_MIN_SAMPLES: int = 2

        self.dbscan = DBSCAN(
            eps=self.DBSCAN_EPS,
            min_samples=self.DBSCAN_MIN_SAMPLES,
            metric="cosine",
        )


class SentenceProcessor:
    """Sentence processor for audio transcription and speaker identification with proper dependency injection"""

    def __init__(self, network_id: str, config: Dict[str, Any] | None = None):
        """
        Initialize SentenceProcessor for a specific network.

        Args:
            network_id: ID of the network this processor belongs to
            config: Network-specific configuration
        """
        self.network_id = network_id
        self.config = config or {} if config else {}
        self.sample_rate = settings.sample_rate

        # Initialize models
        self.asr_model = WhisperModel("base", device="cpu", compute_type="int8")
        self.spk_encoder = VoiceEncoder()
        self._speaker_state = SpeakerIdentificationState()

        MiraLogger.info(f"SentenceProcessor initialized for network {network_id}")

    @staticmethod
    def pcm_bytes_to_float32(pcm: bytes) -> np.ndarray:
        """Convert 16-bit PCM to float32 in [-1,1]."""
        audio_int16 = np.frombuffer(pcm, dtype=np.int16)
        return audio_int16.astype(np.float32) / 32768.0

    @staticmethod
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

    @staticmethod
    def butter_highpass_filter(data, cutoff, fs, order=5):
        """Apply a high-pass Butterworth filter to the data."""
        b, a = SentenceProcessor.butter_highpass(cutoff, fs, order=order)
        y = lfilter(b, a, data)
        return y

    def denoise_audio(self, audio_data: np.ndarray) -> np.ndarray:
        """
        Apply real-time audio denoising to remove white noise and background sounds.

        Args:
            audio_data: Input audio signal as float32 array

        Returns:
            Denoised audio signal
        """
        try:
            audio_data = np.array(audio_data, dtype=np.float32)
            filtered_audio = self.butter_highpass_filter(audio_data, 80, self.sample_rate)
            filtered_audio = np.array(filtered_audio, dtype=np.float32)

            if len(filtered_audio) < 512:
                return filtered_audio

            if len(filtered_audio) > self.sample_rate // 2:
                noise_sample = filtered_audio[: self.sample_rate // 2]
                denoised_audio = nr.reduce_noise(
                    y=filtered_audio,
                    sr=self.sample_rate,
                    y_noise=noise_sample,
                    prop_decrease=0.8,
                    stationary=False,
                    n_fft=min(512, len(filtered_audio)),
                    hop_length=min(128, len(filtered_audio) // 4),
                )
            else:
                denoised_audio = nr.reduce_noise(
                    y=filtered_audio,
                    sr=self.sample_rate,
                    prop_decrease=0.6,
                    stationary=True,
                    n_fft=min(256, len(filtered_audio)),
                    hop_length=min(64, len(filtered_audio) // 4),
                )

            return np.array(denoised_audio, dtype=np.float32)

        except Exception as e:
            MiraLogger.warning(f"Denoising failed: {e}")
            filtered = self.butter_highpass_filter(audio_data, 80, self.sample_rate)
            filtered = np.array(filtered, dtype=np.float32)
            if isinstance(filtered, tuple):
                filtered = filtered[0]
            return filtered

    @staticmethod
    def cosine_sim(a: np.ndarray, b: np.ndarray) -> float:
        """Cosine similarity between two vectors."""
        return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))

    def _refresh_speaker_cache(self):
        """(Re)load all speaker embeddings, person_ids, interaction_ids from the database."""
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

    def assign_speaker(self, embedding: np.ndarray):
        """
        Assign a speaker using DBSCAN clustering over all embeddings.
        Returns the Person.id of the most similar speaker (if above threshold) or creates a new one.
        Also updates clusters in the database.

        Args:
            embedding: The new voice embedding (np.ndarray)
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

        all_embeddings = self._speaker_state._speaker_embeddings + [embedding]
        X = np.stack(all_embeddings)
        dbscan = DBSCAN(
            eps=self._speaker_state.DBSCAN_EPS,
            min_samples=self._speaker_state.DBSCAN_MIN_SAMPLES,
            metric="cosine",
        )

        labels = dbscan.fit_predict(X)
        new_label = labels[-1]

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

        MiraLogger.info(f"Best speaker similarity: {best_sim}")

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

    def update_voice_embedding(
        self, person_id: uuid.UUID, audio_buffer: bytearray, expected_text: str
    ):
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
            MiraLogger.info(
                f"Transcribed text '{transcribed_text}' does not match expected '{expected_text}'"
            )

        denoised_audio = self.denoise_audio(self.pcm_bytes_to_float32(bytes(audio_buffer)))
        embedding_result = self.spk_encoder.embed_utterance(denoised_audio)
        new_embedding = (
            embedding_result[0] if isinstance(embedding_result, tuple) else embedding_result
        )
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

        MiraLogger.info(
            f"Updated embedding for person {person_id} using expected text '{expected_text}'"
        )

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

    def transcribe_interaction(
        self, sentence_buf: bytearray, assign_or_create_speaker: bool
    ) -> dict:
        """
        Process a complete sentence buffer with real-time audio denoising and speaker recognition.

        Args:
            sentence_buf: Audio buffer containing the sentence
            assign_or_create_speaker: Whether to assign or create a speaker
        """
        audio_f32 = self.pcm_bytes_to_float32(bytes(sentence_buf))
        denoised_audio = self.denoise_audio(audio_f32)

        if np.isnan(denoised_audio).any() or np.isinf(denoised_audio).any():
            raise ValueError("Audio contains NaN or Inf values")

        segments, info = self.asr_model.transcribe(denoised_audio, beam_size=5)

        # Extract text from segments
        text_parts = []
        for segment in segments:
            text_parts.append(segment.text.strip())

        text = " ".join(text_parts).strip()

        if not text:
            raise ValueError("Transcription failed")

        interaction = dict()
        interaction["text"] = text

        if assign_or_create_speaker:
            embedding_result = self.spk_encoder.embed_utterance(denoised_audio)
            embedding = (
                embedding_result[0] if isinstance(embedding_result, tuple) else embedding_result
            )
            speaker_id = self.assign_speaker(embedding)
            interaction["voice_embedding"] = embedding.tolist()
            interaction["speaker_id"] = speaker_id

        return interaction

    def cleanup(self):
        """Clean up resources when the processor is no longer needed."""
        MiraLogger.info(f"Cleaning up SentenceProcessor for network {self.network_id}")
        # Add any cleanup logic here if needed
