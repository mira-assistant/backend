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
Now uses proper dependency injection and lifecycle management.
"""

from typing import Any, Dict, List, Optional
from dataclasses import dataclass, field
from datetime import datetime, timezone
import threading
import numpy as np
from scipy import signal
from app.core.mira_logger import MiraLogger
from app.core.config import settings


@dataclass
class StreamQualityMetrics:
    """Container for stream quality metrics"""

    snr: float = 0.0
    speech_clarity: float = 0.0
    volume_level: float = 0.0
    noise_level: float = 0.0
    location: Optional[Dict] = None
    rssi: Optional[float] = None
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    sample_count: int = 0
    score: float = field(init=False, default=0.0)


@dataclass
class ClientStreamInfo:
    """Information about a connected client's audio stream"""

    client_id: str
    last_update: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    quality_metrics: StreamQualityMetrics = field(default_factory=StreamQualityMetrics)


class MultiStreamProcessor:
    """
    Audio Stream Scoring System for evaluating and selecting the best stream.

    This class provides real-time scoring of multiple audio streams based on
    signal quality metrics and dynamically selects the best performing stream.
    Now uses proper dependency injection and lifecycle management.
    """

    def __init__(self, network_id: str, config: Dict[str, Any] | None = None):
        """
        Initialize the audio stream scorer for a specific network.

        Args:
            network_id: ID of the network this processor belongs to
            config: Network-specific configuration
        """
        self.network_id = network_id
        self.config = config or {}
        self.sample_rate = settings.sample_rate
        self.clients: Dict[str, ClientStreamInfo] = {}
        self._lock = threading.Lock()

        # Scoring weights (can be adjusted based on requirements)
        self.weights = {
            "snr": 0.3,
            "speech_clarity": 0.3,
            "volume_level": 0.1,
            "location": 0.15,
            "rssi": 0.15,
        }

        MiraLogger.info(f"MultiStreamProcessor initialized for network {network_id}")

    def register_client(self, client_id: str) -> bool:
        """
        Register a new client for stream scoring.

        Args:
            client_id: Unique identifier for the client

        Returns:
            bool: True if registration successful
        """
        with self._lock:
            if client_id in self.clients:
                MiraLogger.warning(
                    f"Client {client_id} already registered for network {self.network_id}, updating info"
                )

            self.clients[client_id] = ClientStreamInfo(client_id=client_id)

            MiraLogger.info(
                f"Registered client {client_id} for stream scoring in network {self.network_id}"
            )
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
                MiraLogger.warning(
                    f"Client {client_id} not found for deregistration in network {self.network_id}"
                )
                return False

            del self.clients[client_id]

            MiraLogger.info(f"Deregistered client {client_id} from network {self.network_id}")
            return True

    def _calculate_snr(self, audio_data) -> float:
        """
        Calculate Signal-to-Noise Ratio for audio data.

        Args:
            audio_data: Audio signal as numpy array

        Returns:
            float: SNR in dB
        """
        if len(audio_data) == 0:
            return 0.0

        signal_power = np.var(audio_data)

        if signal_power == 0:
            return 0.0

        # Estimate noise using the quieter portions (bottom 20% of audio power)
        # This is a simple noise estimation approach
        windowed_power = []
        window_size = len(audio_data) // 10

        if window_size < 100:
            noise_power = signal_power * 0.1
        else:
            for i in range(0, len(audio_data) - window_size, window_size):
                window = audio_data[i : i + window_size]
                windowed_power.append(np.var(window))

            # Use bottom 20% as noise estimate
            windowed_power.sort()
            noise_power = np.mean(windowed_power[: max(1, len(windowed_power) // 5)])

        if noise_power <= 0:
            noise_power = signal_power * 0.01

        # Calculate SNR in dB
        snr_db = 10 * np.log10(signal_power / noise_power)
        return max(0.0, snr_db)

    def _calculate_speech_clarity(self, audio_data) -> float:
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
                MiraLogger.warning(
                    f"Client {client_id} not registered for quality updates in network {self.network_id}"
                )
                return None

            client_info = self.clients[client_id]

            # Calculate quality metrics
            snr = self._calculate_snr(audio_data)
            speech_clarity = self._calculate_speech_clarity(audio_data)

            volume_level = float(np.sqrt(np.mean(audio_data**2)))  # RMS volume
            noise_level = max(0.0, volume_level - (snr / 20.0))  # Estimate based on SNR

            # Update metrics
            metrics = StreamQualityMetrics(
                snr=snr,
                speech_clarity=speech_clarity,
                volume_level=volume_level,
                noise_level=noise_level,
                location=client_info.quality_metrics.location,  # Preserve existing value
                rssi=client_info.quality_metrics.rssi,  # Preserve existing value
                sample_count=client_info.quality_metrics.sample_count + 1,
            )

            client_info.quality_metrics = metrics
            client_info.last_update = datetime.now(timezone.utc)

            MiraLogger.debug(
                f"Updated quality for {client_id} in network {self.network_id}: SNR={snr:.1f}dB, Clarity={speech_clarity:.1f}"
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

        # Location score (will be 100 if no location info)
        location_score = 100.0
        if metrics.location is not None:
            # Simple scoring based on location accuracy - better accuracy = higher score
            accuracy = metrics.location.get("accuracy", 100.0)  # meters
            location_score = max(0.0, 100.0 - (accuracy / 10.0))  # Better accuracy = higher score

        # RSSI score (will be 100 if no RSSI info)
        rssi_score = 100.0
        if metrics.rssi is not None:
            # Higher RSSI (less negative) = better signal = higher score
            # Typical RSSI range: -30 (excellent) to -90 (poor)
            rssi_normalized = max(-90.0, min(-30.0, metrics.rssi))  # Clamp to typical range
            rssi_score = ((rssi_normalized + 90.0) / 60.0) * 100.0  # Convert to 0-100

        # Calculate weighted score
        overall_score = (
            self.weights["snr"] * snr_score
            + self.weights["speech_clarity"] * clarity_score
            + self.weights["volume_level"] * volume_score
            + self.weights["location"] * location_score
            + self.weights["rssi"] * rssi_score
        )

        return min(100.0, max(0.0, overall_score))

    def get_best_stream(self) -> Dict[str, float]:
        """
        Get the client ID with the best current stream quality.

        Returns:
            Dict[str, float]: {"client_id": client_id, "score": score} of best stream
        """
        best_stream = {
            "client_id": None,
            "score": 0.0,
        }

        for client_id, client_info in self.clients.items():
            # Calculate current score for this client
            current_score = self.calculate_overall_score(client_info.quality_metrics)
            client_info.quality_metrics.score = current_score

            if current_score > best_stream["score"]:
                best_stream = {"client_id": client_id, "score": current_score}

        return best_stream

    def get_all_stream_scores(self) -> Dict[str, float]:
        """
        Get current scores for all active streams.

        Returns:
            Dict[str, float]: Mapping of client_id to current score
        """
        with self._lock:
            scores = {}
            for client_id, client_info in self.clients.items():
                score = self.calculate_overall_score(client_info.quality_metrics)

                scores[client_id] = score
                client_info.quality_metrics.score = score
            return scores

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
                    del self.clients[client_id]

                    removed_clients.append(client_id)
                    MiraLogger.info(
                        f"Removed inactive client {client_id} from network {self.network_id}"
                    )

            return removed_clients

    def set_phone_location(self, client_id: str, location: Dict) -> bool:
        """
        Set GPS location data for a client.

        Args:
            client_id: Unique identifier for the client
            location: GPS location data (lat, lng, accuracy, etc.)

        Returns:
            bool: True if location was set successfully
        """
        with self._lock:
            if client_id not in self.clients:
                MiraLogger.warning(
                    f"Client {client_id} not found for location update in network {self.network_id}"
                )
                return False

            self.clients[client_id].quality_metrics.location = location
            MiraLogger.info(
                f"Set location for {client_id} in network {self.network_id}: {location}"
            )
            return True

    def set_phone_rssi(self, client_id: str, rssi: float) -> bool:
        """
        Set RSSI signal strength for a client.

        Args:
            client_id: Unique identifier for the client
            rssi: RSSI signal strength in dBm

        Returns:
            bool: True if RSSI was set successfully
        """
        with self._lock:
            if client_id not in self.clients:
                MiraLogger.warning(
                    f"Client {client_id} not found for RSSI update in network {self.network_id}"
                )
                return False

            self.clients[client_id].quality_metrics.rssi = rssi
            MiraLogger.info(f"Set RSSI for {client_id} in network {self.network_id}: {rssi} dBm")
            return True

    def cleanup(self):
        """Clean up resources when the processor is no longer needed."""
        MiraLogger.info(f"Cleaning up MultiStreamProcessor for network {self.network_id}")
        # Clean up any resources if needed
        with self._lock:
            self.clients.clear()
