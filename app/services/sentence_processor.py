"""
Live interaction with real-time speaker diarization (up to two speakers) using
local Whisper and pyannote.audio with real-time audio denoising.

Supports multilingual speech recognition including English and Indian languages.
Code-mixed speech (e.g., English + Hindi/Tamil/etc.) is automatically detected.
Real-time audio denoising filters out white noise and background sounds.

Now uses proper dependency injection and lifecycle management.
"""

from __future__ import annotations

import os
import uuid
from typing import Any, Dict, Optional

import noisereduce as nr
import numpy as np
import torch
from faster_whisper import WhisperModel
from pyannote.audio import Pipeline
from scipy.signal import butter, lfilter
from sqlalchemy.orm import Session

from core.constants import CONTEXT_SIMILARITY_THRESHOLD, SAMPLE_RATE
from core.mira_logger import MiraLogger
from db import get_db_session
from models import Interaction, Person

# MiraLogger is used directly via class methods


class SentenceProcessor:
    """
    Sentence processor for audio transcription and speaker identification
    with proper dependency injection
    """

    def __init__(self, network_id: str, config: Dict[str, Any] | None = None):
        """
        Initialize SentenceProcessor for a specific network.

        Args:
            network_id: ID of the network this processor belongs to
            config: Network-specific configuration
        """
        self.network_id = network_id
        self.config = config or {} if config else {}
        self.sample_rate = SAMPLE_RATE

        # Initialize models
        self.asr_model = WhisperModel("small", device="cpu", compute_type="int8")

        # Initialize pyannote.audio diarization pipeline
        self._initialize_speaker_diarization()

        # Initialize speaker cache for consistency with existing database structure
        self._speaker_embeddings: list[np.ndarray] = []
        self._speaker_ids: list[uuid.UUID] = []
        self._clusters_dirty: bool = True

        MiraLogger.info(f"SentenceProcessor initialized for network {network_id}")

    def _initialize_speaker_diarization(self):
        """Initialize the pyannote.audio speaker diarization pipeline."""
        try:
            # Check for Hugging Face token
            hf_token = os.getenv("HUGGING_FACE_TOKEN")
            if not hf_token:
                MiraLogger.warning(
                    "HUGGING_FACE_TOKEN environment variable not set. "
                    "Speaker diarization will use fallback mode."
                )
                self.diarization_pipeline = None
                return

            # Load the pre-trained speaker diarization pipeline
            self.diarization_pipeline = Pipeline.from_pretrained(
                "pyannote/speaker-diarization-cambridge", use_auth_token=hf_token
            )

            # Set device to CPU for compatibility
            if torch.cuda.is_available():
                self.diarization_pipeline.to(torch.device("cuda"))
            else:
                self.diarization_pipeline.to(torch.device("cpu"))

            MiraLogger.info(
                "Pyannote.audio speaker diarization pipeline loaded successfully"
            )

        except Exception as e:
            MiraLogger.warning(
                f"Failed to initialize speaker diarization, using fallback mode: {e}"
            )
            # Fallback: create a mock pipeline to maintain API compatibility
            self.diarization_pipeline = None

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
                f"Cutoff frequency must be between 0 and Nyquist "
                f"({nyquist} Hz), got {cutoff}"
            )
        normal_cutoff = cutoff / nyquist
        result = butter(order, normal_cutoff, btype="high", analog=False)
        if result is None:
            raise ValueError(
                "Butterworth filter design failed; check cutoff frequency."
            )
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
            filtered_audio = self.butter_highpass_filter(
                audio_data, 80, self.sample_rate
            )
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

    def transcribe_interaction(
        self, sentence_buf: bytearray, assign_or_create_speaker: bool
    ) -> dict:
        """
        Process a complete sentence buffer with real-time audio denoising
        and speaker recognition.

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
            if self.diarization_pipeline is not None:
                try:
                    # Use pyannote.audio for speaker diarization
                    speaker_id, embedding = self._identify_speaker(denoised_audio)
                    interaction["voice_embedding"] = (
                        embedding.tolist() if embedding is not None else None
                    )
                    interaction["speaker_id"] = speaker_id
                except Exception as e:
                    MiraLogger.warning(f"Speaker identification failed: {e}")
                    # Fallback: assign a default speaker if identification fails
                    interaction["speaker_id"] = self._get_or_create_default_speaker()
            else:
                # Fallback mode: create simple hash-based speaker identification
                MiraLogger.info(
                    "Using fallback speaker identification (no pyannote.audio)"
                )
                speaker_id, embedding = self._fallback_speaker_identification(
                    denoised_audio
                )
                interaction["voice_embedding"] = (
                    embedding.tolist() if embedding is not None else None
                )
                interaction["speaker_id"] = speaker_id

        return interaction

    def _fallback_speaker_identification(
        self, audio_data: np.ndarray
    ) -> tuple[uuid.UUID, np.ndarray]:
        """
        Fallback speaker identification when pyannote.audio is not available.
        Uses basic audio characteristics to create a consistent speaker embedding.

        Args:
            audio_data: Audio data as numpy array

        Returns:
            Tuple of (speaker_id, embedding)
        """
        # Create a basic embedding based on audio characteristics
        import hashlib

        # Extract basic audio features
        audio_features = np.array(
            [
                np.mean(audio_data),
                np.std(audio_data),
                np.max(audio_data),
                np.min(audio_data),
                len(audio_data),
                np.mean(np.abs(np.diff(audio_data))),  # roughness
                np.percentile(audio_data, 25),  # quartiles
                np.percentile(audio_data, 75),
            ]
        )

        # Create a hash from the audio features for consistency
        feature_str = ",".join([f"{f:.6f}" for f in audio_features])
        feature_hash = hashlib.md5(feature_str.encode()).hexdigest()

        # Generate a consistent 256-dimensional embedding
        np.random.seed(int(feature_hash[:8], 16))
        embedding = np.random.randn(256).astype(np.float32)

        # Incorporate actual audio features
        embedding[: len(audio_features)] = audio_features

        # Normalize
        embedding = embedding / np.linalg.norm(embedding)

        # Find or create speaker based on this embedding
        speaker_id = self._find_or_create_speaker(embedding)

        return speaker_id, embedding

    def _identify_speaker(
        self, audio_data: np.ndarray
    ) -> tuple[Optional[uuid.UUID], Optional[np.ndarray]]:
        """
        Identify speaker using pyannote.audio pipeline.

        Args:
            audio_data: Audio data as numpy array

        Returns:
            Tuple of (speaker_id, embedding) where speaker_id is the Person.id
            and embedding is the speaker embedding
        """
        if self.diarization_pipeline is None:
            return None, None

        try:
            # Convert numpy array to pyannote-compatible format
            # Create a temporary audio structure for pyannote
            import tempfile
            import soundfile as sf

            # Save audio temporarily for pyannote processing
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp_file:
                sf.write(tmp_file.name, audio_data, self.sample_rate)

                # Perform diarization
                diarization = self.diarization_pipeline(tmp_file.name)

                # Extract speaker embeddings and find the most prominent speaker
                speaker_labels = list(diarization.labels())

                if not speaker_labels:
                    # No speakers detected, create a new one
                    return self._create_new_speaker(), None

                # For now, take the first speaker (most prominent)
                # In a more sophisticated implementation, we could analyze which speaker
                # has the most speech time or confidence
                primary_speaker_label = speaker_labels[0]

                # Get embedding for this speaker segment
                # For simplicity, we'll use a synthetic embedding based on
                # the speaker label. In a production system, you'd extract
                # actual embeddings from the pipeline
                synthetic_embedding = self._generate_speaker_embedding(
                    primary_speaker_label, audio_data
                )

                # Find or create speaker based on embedding similarity
                speaker_id = self._find_or_create_speaker(synthetic_embedding)

                # Clean up temporary file
                os.unlink(tmp_file.name)

                return speaker_id, synthetic_embedding

        except Exception as e:
            MiraLogger.error(f"Error in speaker identification: {e}")
            return self._create_new_speaker(), None

    def _generate_speaker_embedding(
        self, speaker_label: str, audio_data: np.ndarray
    ) -> np.ndarray:
        """
        Generate a speaker embedding from audio data.
        This is a simplified implementation - in production you'd use the actual
        embedding extraction from pyannote.audio.

        Args:
            speaker_label: Speaker label from diarization
            audio_data: Audio data

        Returns:
            Speaker embedding as numpy array
        """
        # For now, create a hash-based embedding from the speaker characteristics
        # This is a placeholder - in production, you'd use actual neural embeddings
        import hashlib

        # Create a basic embedding based on audio features and speaker label
        audio_stats = np.array(
            [
                np.mean(audio_data),
                np.std(audio_data),
                np.max(audio_data),
                np.min(audio_data),
                len(audio_data),
            ]
        )

        # Combine with speaker label hash for consistency
        label_hash = int(hashlib.md5(speaker_label.encode()).hexdigest()[:8], 16)

        # Create a 256-dimensional embedding (similar to resemblyzer)
        embedding = np.random.RandomState(label_hash).randn(256).astype(np.float32)

        # Add some audio-derived features
        embedding[:5] = audio_stats

        # Normalize
        embedding = embedding / np.linalg.norm(embedding)

        return embedding

    def _find_or_create_speaker(self, embedding: np.ndarray) -> uuid.UUID:
        """
        Find existing speaker by embedding similarity or create a new one.

        Args:
            embedding: Speaker embedding

        Returns:
            Speaker ID (Person.id)
        """
        session = get_db_session()
        try:
            # Refresh speaker cache if needed
            if self._clusters_dirty or not self._speaker_embeddings:
                self._refresh_speaker_cache()

            # Find most similar speaker
            best_similarity = -1.0
            best_speaker_id = None

            for i, cached_embedding in enumerate(self._speaker_embeddings):
                similarity = self.cosine_sim(embedding, cached_embedding)
                if similarity > best_similarity:
                    best_similarity = similarity
                    best_speaker_id = self._speaker_ids[i]

            # If similarity is above threshold, return existing speaker
            if (
                best_similarity > CONTEXT_SIMILARITY_THRESHOLD
                and best_speaker_id is not None
            ):
                # Update the embedding with exponential moving average
                self._update_speaker_embedding(best_speaker_id, embedding, session)
                return best_speaker_id

            # Create new speaker
            return self._create_new_speaker_with_embedding(embedding, session)

        finally:
            session.close()

    def _create_new_speaker(self) -> uuid.UUID:
        """Create a new speaker without embedding."""
        session = get_db_session()
        try:
            return self._create_new_speaker_with_embedding(None, session)
        finally:
            session.close()

    def _create_new_speaker_with_embedding(
        self, embedding: Optional[np.ndarray], session: Session
    ) -> uuid.UUID:
        """Create a new speaker with optional embedding."""
        # Get next index
        next_index = (
            session.query(Person.index).order_by(Person.index.desc()).first() or [0]
        )[0] + 1

        # Create new person
        new_person = Person(
            voice_embedding=embedding.tolist() if embedding is not None else None,
            index=next_index,
            network_id=(
                uuid.UUID(self.network_id)
                if isinstance(self.network_id, str)
                else self.network_id
            ),
        )

        session.add(new_person)
        session.commit()
        session.refresh(new_person)

        # Update cache
        if embedding is not None:
            self._speaker_embeddings.append(embedding)
            self._speaker_ids.append(new_person.id)
            self._clusters_dirty = True

        MiraLogger.info(
            f"Created new speaker with ID {new_person.id} and index {next_index}"
        )
        return new_person.id

    def _update_speaker_embedding(
        self, speaker_id: uuid.UUID, new_embedding: np.ndarray, session: Session
    ):
        """Update existing speaker embedding with exponential moving average."""
        person = session.query(Person).filter_by(id=speaker_id).first()
        if person and person.voice_embedding is not None:
            old_embedding = np.array(person.voice_embedding, dtype=np.float32)
            # Exponential moving average: 80% old, 20% new
            updated_embedding = 0.8 * old_embedding + 0.2 * new_embedding
            person.voice_embedding = updated_embedding.tolist()
            session.commit()

            # Update cache
            if speaker_id in self._speaker_ids:
                idx = self._speaker_ids.index(speaker_id)
                self._speaker_embeddings[idx] = updated_embedding

    def _get_or_create_default_speaker(self) -> uuid.UUID:
        """Get or create a default speaker for fallback cases."""
        session = get_db_session()
        try:
            # Try to find a default speaker
            default_speaker = session.query(Person).filter_by(index=1).first()
            if default_speaker:
                return default_speaker.id

            # Create default speaker if none exists
            return self._create_new_speaker_with_embedding(None, session)
        finally:
            session.close()

    def _refresh_speaker_cache(self):
        """
        (Re)load all speaker embeddings, person_ids, interaction_ids from
        the database.
        """
        session = get_db_session()
        try:
            interactions = (
                session.query(Interaction)
                .filter(
                    Interaction.voice_embedding.isnot(None),
                    Interaction.speaker_id.isnot(None),
                )
                .all()
            )
            self._speaker_embeddings = [
                np.array(i.voice_embedding, dtype=np.float32) for i in interactions
            ]
            self._speaker_ids = [i.speaker_id for i in interactions]  # type: ignore
            self._clusters_dirty = False
        finally:
            session.close()

    def cleanup(self):
        """Clean up resources when the processor is no longer needed."""
        MiraLogger.info(f"Cleaning up SentenceProcessor for network {self.network_id}")
        # Add any cleanup logic here if needed
        # Add any cleanup logic here if needed
