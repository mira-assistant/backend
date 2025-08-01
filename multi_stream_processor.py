"""
Audio Stream Scoring System for Dynamic Stream Selection

This module implements a scoring system to evaluate and select the best audio stream
from multiple connected listening clients based on signal quality metrics.

The scoring system evaluates streams based on:
- Signal-to-Noise Ratio (SNR)
- Speech clarity metrics
- Future placeholder for phone distance consideration

The system is designed for real-time operation and can dynamically select
the best audio stream for optimal recording quality.
"""

import logging
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field
from datetime import datetime, timezone
import threading
import numpy as np
from scipy import signal


logger = logging.getLogger(__name__)


@dataclass
class StreamQualityMetrics:
    """Container for stream quality metrics"""

    snr: float = 0.0
    speech_clarity: float = 0.0
    volume_level: float = 0.0
    noise_level: float = 0.0
    # Placeholder for future phone distance feature
    phone_distance: Optional[float] = None
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    sample_count: int = 0


@dataclass
class ClientStreamInfo:
    """Information about a connected client's audio stream"""

    client_id: str
    is_active: bool = True
    last_update: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    quality_metrics: StreamQualityMetrics = field(default_factory=StreamQualityMetrics)
    # Metadata for future features
    device_type: Optional[str] = None
    location: Optional[Dict] = None  # For future distance calculation


class AudioStreamScorer:
    """
    Audio Stream Scoring System for evaluating and selecting the best stream.

    This class provides real-time scoring of multiple audio streams based on
    signal quality metrics and dynamically selects the best performing stream.
    """

    def __init__(self, sample_rate: int = 16000):
        """
        Initialize the audio stream scorer.

        Args:
            sample_rate: Audio sample rate for processing
        """
        self.sample_rate = sample_rate
        self.clients: Dict[str, ClientStreamInfo] = {}
        self.current_best_client: Optional[str] = None
        self.score_history: Dict[str, List[float]] = {}
        self._lock = threading.Lock()

        # Scoring weights (can be adjusted based on requirements)
        self.weights = {
            "snr": 0.4,
            "speech_clarity": 0.4,
            "volume_level": 0.1,
            "phone_distance": 0.1,  # Placeholder for future use
        }

    def register_client(
        self, client_id: str, device_type: Optional[str] = None, location: Optional[Dict] = None
    ) -> bool:
        """
        Register a new client for stream scoring.

        Args:
            client_id: Unique identifier for the client
            device_type: Type of device (phone, tablet, etc.) - for future use
            location: Location information for distance calculation - for future use

        Returns:
            bool: True if registration successful
        """
        with self._lock:
            if client_id in self.clients:
                logger.warning(f"Client {client_id} already registered, updating info")

            self.clients[client_id] = ClientStreamInfo(
                client_id=client_id, device_type=device_type, location=location
            )
            self.score_history[client_id] = []

            logger.info(f"Registered client {client_id} for stream scoring")
            return True

    def deregister_client(self, client_id: str) -> bool:
        """
        Deregister a client from stream scoring.

        Args:
            client_id: Unique identifier for the client

        Returns:
            bool: True if deregistration successful
        """
        with self._lock:
            if client_id not in self.clients:
                logger.warning(f"Client {client_id} not found for deregistration")
                return False

            # If this was the best client, clear the selection
            if self.current_best_client == client_id:
                self.current_best_client = None
                logger.info(f"Best client {client_id} deregistered, clearing selection")

            del self.clients[client_id]
            if client_id in self.score_history:
                del self.score_history[client_id]

            logger.info(f"Deregistered client {client_id}")
            return True

    def calculate_snr(self, audio_data) -> float:
        """
        Calculate Signal-to-Noise Ratio for audio data.

        Args:
            audio_data: Audio signal as numpy array

        Returns:
            float: SNR in dB
        """

        if len(audio_data) == 0:
            return 0.0

        # Calculate signal power (using the variance of the signal)
        signal_power = np.var(audio_data)

        if signal_power == 0:
            return 0.0

        # Estimate noise using the quieter portions (bottom 20% of audio power)
        # This is a simple noise estimation approach
        windowed_power = []
        window_size = len(audio_data) // 10

        if window_size < 100:  # Too small for windowing
            noise_power = signal_power * 0.1  # Assume 10% noise
        else:
            for i in range(0, len(audio_data) - window_size, window_size):
                window = audio_data[i : i + window_size]
                windowed_power.append(np.var(window))

            # Use bottom 20% as noise estimate
            windowed_power.sort()
            noise_power = np.mean(windowed_power[: max(1, len(windowed_power) // 5)])

        if noise_power <= 0:
            noise_power = signal_power * 0.01  # Fallback to 1% of signal

        # Calculate SNR in dB
        snr_db = 10 * np.log10(signal_power / noise_power)
        return max(0.0, snr_db)  # Ensure non-negative

    def calculate_speech_clarity(self, audio_data) -> float:
        """
        Calculate speech clarity metric based on spectral analysis.

        Args:
            audio_data: Audio signal as numpy array

        Returns:
            float: Speech clarity score (0-100)
        """

        if len(audio_data) == 0:
            return 0.0

        # Calculate power spectral density
        # Use smaller nperseg for short audio clips
        nperseg = min(1024, len(audio_data) // 4) if len(audio_data) > 256 else len(audio_data) // 2
        nperseg = max(nperseg, 64)  # Minimum window size
        nperseg = min(nperseg, len(audio_data))  # Ensure nperseg does not exceed audio length

        freqs, psd = signal.welch(audio_data, fs=self.sample_rate, nperseg=nperseg)

        # Speech frequency ranges (fundamental frequencies for clarity)
        # Focus on 300-3400 Hz range which is most important for speech intelligibility
        speech_low = 300
        speech_high = 3400

        speech_mask = (freqs >= speech_low) & (freqs <= speech_high)
        speech_power = np.sum(psd[speech_mask])
        total_power = np.sum(psd)

        if total_power == 0:
            return 0.0

        # Speech clarity as ratio of speech-band power to total power
        speech_ratio = speech_power / total_power

        # Also consider spectral flatness (measure of how noise-like vs. tonal)
        # Lower spectral flatness in speech band indicates clearer speech
        if np.any(psd[speech_mask] > 0):
            geometric_mean = np.exp(np.mean(np.log(psd[speech_mask] + 1e-10)))
            arithmetic_mean = np.mean(psd[speech_mask])
            spectral_flatness = geometric_mean / (arithmetic_mean + 1e-10)
            clarity_factor = 1.0 - spectral_flatness  # Invert so higher is better
        else:
            clarity_factor = 0.0

        # Combine speech ratio and clarity factor
        clarity_score = (speech_ratio * 0.7 + clarity_factor * 0.3) * 100
        return min(100.0, max(0.0, float(clarity_score)))

    def update_stream_quality(self, client_id: str, audio_data) -> Optional[StreamQualityMetrics]:
        """
        Update quality metrics for a client's audio stream.

        Args:
            client_id: Unique identifier for the client
            audio_data: Raw audio data as numpy array

        Returns:
            StreamQualityMetrics: Updated metrics for the stream, or None if client not found
        """
        with self._lock:
            if client_id not in self.clients:
                logger.warning(f"Client {client_id} not registered for quality updates")
                return None

            client_info = self.clients[client_id]

            # Calculate quality metrics
            snr = self.calculate_snr(audio_data)
            speech_clarity = self.calculate_speech_clarity(audio_data)

            volume_level = float(np.sqrt(np.mean(audio_data**2)))  # RMS volume
            noise_level = max(0.0, volume_level - (snr / 20.0))  # Estimate based on SNR

            # Update metrics
            metrics = StreamQualityMetrics(
                snr=snr,
                speech_clarity=speech_clarity,
                volume_level=volume_level,
                noise_level=noise_level,
                phone_distance=client_info.quality_metrics.phone_distance,  # Preserve existing value
                sample_count=client_info.quality_metrics.sample_count + 1,
            )

            client_info.quality_metrics = metrics
            client_info.last_update = datetime.now(timezone.utc)
            client_info.is_active = True

            logger.debug(
                f"Updated quality for {client_id}: SNR={snr:.1f}dB, Clarity={speech_clarity:.1f}"
            )
            return metrics

    def calculate_overall_score(self, metrics: StreamQualityMetrics) -> float:
        """
        Calculate overall quality score for a stream.

        Args:
            metrics: Stream quality metrics

        Returns:
            float: Overall score (0-100)
        """
        # Normalize individual metrics to 0-100 scale
        snr_score = min(100.0, (metrics.snr / 30.0) * 100)  # Assume 30dB is excellent
        clarity_score = metrics.speech_clarity  # Already 0-100
        volume_score = min(100.0, metrics.volume_level * 1000)  # Scale volume appropriately

        # Distance score placeholder (will be 100 if no distance info)
        distance_score = 100.0
        if metrics.phone_distance is not None:
            # Closer distance = higher score (future implementation)
            # For now, just placeholder logic
            distance_score = max(0.0, 100.0 - (metrics.phone_distance * 10))

        # Calculate weighted score
        overall_score = (
            self.weights["snr"] * snr_score
            + self.weights["speech_clarity"] * clarity_score
            + self.weights["volume_level"] * volume_score
            + self.weights["phone_distance"] * distance_score
        )

        return min(100.0, max(0.0, overall_score))

    def get_best_stream(self) -> Optional[Tuple[str, float]]:
        """
        Get the client ID with the best current stream quality.

        Returns:
            Tuple[str, float]: (client_id, score) of best stream, or None if no active clients
        """
        with self._lock:
            if not self.clients:
                return None

            # Optimization: If only one active client, return it immediately without scoring
            active_clients = [
                client_id
                for client_id, client_info in self.clients.items()
                if client_info.is_active
            ]

            if len(active_clients) == 1:
                single_client = active_clients[0]
                if single_client != self.current_best_client:
                    self.current_best_client = single_client
                    logger.info(f"Single active client: {single_client} (no scoring needed)")
                # Return a default score for single client
                return (single_client, 1.0)

            best_client = None
            best_score = -1.0

            for client_id, client_info in self.clients.items():
                if not client_info.is_active:
                    continue

                score = self.calculate_overall_score(client_info.quality_metrics)

                # Store score history
                if client_id not in self.score_history:
                    self.score_history[client_id] = []
                self.score_history[client_id].append(score)

                # Keep only recent history (last 10 scores)
                if len(self.score_history[client_id]) > 10:
                    self.score_history[client_id] = self.score_history[client_id][-10:]

                if score > best_score:
                    best_score = score
                    best_client = client_id

            # Update current best client
            if best_client != self.current_best_client:
                old_best = self.current_best_client
                self.current_best_client = best_client
                logger.info(
                    f"Best stream changed from {old_best} to {best_client} (score: {best_score:.1f})"
                )

            return (best_client, best_score) if best_client else None

    def get_all_stream_scores(self) -> Dict[str, float]:
        """
        Get current scores for all active streams.

        Returns:
            Dict[str, float]: Mapping of client_id to current score
        """
        with self._lock:
            scores = {}
            for client_id, client_info in self.clients.items():
                if client_info.is_active:
                    scores[client_id] = self.calculate_overall_score(client_info.quality_metrics)
            return scores

    def set_phone_distance(self, client_id: str, distance: float) -> bool:
        """
        Set phone distance for a client (future feature placeholder).

        Args:
            client_id: Unique identifier for the client
            distance: Distance in meters

        Returns:
            bool: True if distance was set successfully
        """
        with self._lock:
            if client_id not in self.clients:
                logger.warning(f"Client {client_id} not found for distance update")
                return False

            self.clients[client_id].quality_metrics.phone_distance = distance
            logger.info(f"Set phone distance for {client_id}: {distance}m")
            return True

    def get_client_info(self, client_id: str) -> Optional[ClientStreamInfo]:
        """
        Get detailed information about a specific client.

        Args:
            client_id: Unique identifier for the client

        Returns:
            ClientStreamInfo: Client information, or None if not found
        """
        with self._lock:
            return self.clients.get(client_id)

    def cleanup_inactive_clients(self, timeout_seconds: int = 300) -> List[str]:
        """
        Remove clients that haven't been updated recently.

        Args:
            timeout_seconds: Seconds after which to consider a client inactive

        Returns:
            List[str]: List of removed client IDs
        """
        with self._lock:
            current_time = datetime.now(timezone.utc)
            removed_clients = []

            for client_id, client_info in list(self.clients.items()):
                time_diff = (current_time - client_info.last_update).total_seconds()
                if time_diff > timeout_seconds:
                    # Manually remove without calling deregister_client to avoid deadlock
                    if self.current_best_client == client_id:
                        self.current_best_client = None
                        logger.info(f"Best client {client_id} timed out, clearing selection")

                    del self.clients[client_id]
                    if client_id in self.score_history:
                        del self.score_history[client_id]

                    removed_clients.append(client_id)
                    logger.info(f"Removed inactive client {client_id}")

            return removed_clients
