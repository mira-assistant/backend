"""
Tests for the Audio Stream Scoring System
"""

import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import pytest
from datetime import datetime, timezone
import numpy as np

from multi_stream_processor import AudioStreamScorer, StreamQualityMetrics, ClientStreamInfo


class TestAudioStreamScorer:
    """Test suite for AudioStreamScorer class"""

    def setup_method(self):
        """Set up test fixtures"""
        self.scorer = AudioStreamScorer(sample_rate=16000)

    def test_initialization(self):
        """Test scorer initialization"""
        assert self.scorer.sample_rate == 16000
        assert len(self.scorer.clients) == 0
        assert self.scorer.current_best_client is None
        assert len(self.scorer.score_history) == 0

    def test_register_client(self):
        """Test client registration"""
        client_id = "test_client_1"
        success = self.scorer.register_client(client_id, device_type="phone")

        assert success is True
        assert client_id in self.scorer.clients
        assert client_id in self.scorer.score_history
        assert self.scorer.clients[client_id].device_type == "phone"
        assert self.scorer.clients[client_id].is_active is True

    def test_register_duplicate_client(self):
        """Test registering the same client twice"""
        client_id = "test_client_1"

        # First registration
        success1 = self.scorer.register_client(client_id, device_type="phone")
        assert success1 is True

        # Second registration (should update existing)
        success2 = self.scorer.register_client(client_id, device_type="tablet")
        assert success2 is True
        assert self.scorer.clients[client_id].device_type == "tablet"

    def test_deregister_client(self):
        """Test client deregistration"""
        client_id = "test_client_1"

        # Register first
        self.scorer.register_client(client_id)
        assert client_id in self.scorer.clients

        # Then deregister
        success = self.scorer.deregister_client(client_id)
        assert success is True
        assert client_id not in self.scorer.clients
        assert client_id not in self.scorer.score_history

    def test_deregister_nonexistent_client(self):
        """Test deregistering a client that doesn't exist"""
        success = self.scorer.deregister_client("nonexistent_client")
        assert success is False

    def test_deregister_best_client(self):
        """Test deregistering the currently best client"""
        client_id = "test_client_1"

        # Register and set as best client
        self.scorer.register_client(client_id)
        self.scorer.current_best_client = client_id

        # Deregister
        success = self.scorer.deregister_client(client_id)
        assert success is True
        assert self.scorer.current_best_client is None

    def test_calculate_snr_empty_audio(self):
        """Test SNR calculation with empty audio"""
        empty_audio = np.array([])
        snr = self.scorer.calculate_snr(empty_audio)
        assert snr == 0.0

    def test_calculate_snr_silent_audio(self):
        """Test SNR calculation with silent audio"""
        silent_audio = np.zeros(1000)
        snr = self.scorer.calculate_snr(silent_audio)
        assert snr == 0.0

    def test_calculate_snr_noisy_audio(self):
        """Test SNR calculation with noisy audio"""

        # Generate test audio with signal + noise
        sample_rate = 16000
        duration = 1.0  # 1 second
        t = np.linspace(0, duration, int(sample_rate * duration))

        # Pure tone at 440 Hz
        signal = 0.5 * np.sin(2 * np.pi * 440 * t)

        # Add some noise
        noise = 0.1 * np.random.normal(0, 1, len(signal))
        noisy_audio = signal + noise

        snr = self.scorer.calculate_snr(noisy_audio)
        assert snr > 0.0  # Should have positive SNR
        assert snr < 50.0  # Reasonable upper bound

    def test_calculate_speech_clarity_empty_audio(self):
        """Test speech clarity calculation with empty audio"""
        empty_audio = np.array([])
        clarity = self.scorer.calculate_speech_clarity(empty_audio)
        assert clarity == 0.0

    def test_calculate_speech_clarity_valid_audio(self):
        """Test speech clarity calculation with valid audio"""

        # Generate test audio in speech frequency range
        sample_rate = 16000
        duration = 1.0
        t = np.linspace(0, duration, int(sample_rate * duration))

        # Mix of frequencies in speech range (300-3400 Hz)
        speech_audio = (
            0.3 * np.sin(2 * np.pi * 300 * t)
            + 0.4 * np.sin(2 * np.pi * 1000 * t)
            + 0.3 * np.sin(2 * np.pi * 2000 * t)
        )

        clarity = self.scorer.calculate_speech_clarity(speech_audio)
        assert 0.0 <= clarity <= 100.0
        assert clarity > 0.0  # Should have some speech clarity

    def test_update_stream_quality_nonexistent_client(self):
        """Test updating quality for non-existent client"""
        audio_data = np.random.normal(0, 0.1, 1000)
        metrics = self.scorer.update_stream_quality("nonexistent_client", audio_data)
        assert metrics is None

    def test_update_stream_quality_valid_client(self):
        """Test updating quality for valid client"""
        client_id = "test_client_1"
        self.scorer.register_client(client_id)

        # Generate test audio
        audio_data = np.random.normal(0, 0.1, 1000)

        metrics = self.scorer.update_stream_quality(client_id, audio_data)

        assert metrics is not None
        assert isinstance(metrics, StreamQualityMetrics)
        assert metrics.snr >= 0.0
        assert 0.0 <= metrics.speech_clarity <= 100.0
        assert metrics.volume_level >= 0.0
        assert metrics.noise_level >= 0.0
        assert metrics.sample_count == 1

        # Check that client info was updated
        client_info = self.scorer.get_client_info(client_id)

        assert client_info is not None
        assert client_info.is_active is True
        assert client_info.quality_metrics == metrics

    def test_calculate_overall_score(self):
        """Test overall score calculation"""
        metrics = StreamQualityMetrics(
            snr=20.0,  # Good SNR
            speech_clarity=75.0,  # Good clarity
            volume_level=0.1,  # Reasonable volume
            noise_level=0.01,  # Low noise
            phone_distance=None,  # No distance info
        )

        score = self.scorer.calculate_overall_score(metrics)

        assert 0.0 <= score <= 100.0
        assert score > 0.0  # Should have positive score

    def test_calculate_overall_score_with_distance(self):
        """Test overall score calculation with phone distance"""
        metrics = StreamQualityMetrics(
            snr=20.0,
            speech_clarity=75.0,
            volume_level=0.1,
            noise_level=0.01,
            phone_distance=2.0,  # 2 meters
        )

        score = self.scorer.calculate_overall_score(metrics)

        assert 0.0 <= score <= 100.0
        assert score > 0.0

    def test_get_best_stream_no_clients(self):
        """Test getting best stream when no clients are registered"""
        result = self.scorer.get_best_stream()
        assert result is None

    def test_get_best_stream_single_client(self):
        """Test getting best stream with single client"""
        client_id = "test_client_1"
        self.scorer.register_client(client_id)

        # Update with some audio data
        audio_data = np.random.normal(0, 0.1, 1000)
        self.scorer.update_stream_quality(client_id, audio_data)

        result = self.scorer.get_best_stream()

        assert result is not None
        best_client, score = result
        assert best_client == client_id
        assert 0.0 <= score <= 100.0
        assert self.scorer.current_best_client == client_id

    def test_get_best_stream_multiple_clients(self):
        """Test getting best stream with multiple clients"""
        # Register multiple clients
        clients = ["client_1", "client_2", "client_3"]
        for client in clients:
            self.scorer.register_client(client)

        # Update with different quality audio
        for i, client in enumerate(clients):
            # Create audio with different SNR levels
            signal_level = 0.1 + (i * 0.05)  # Increasing signal levels
            audio_data = np.random.normal(0, signal_level, 1000)
            self.scorer.update_stream_quality(client, audio_data)

        result = self.scorer.get_best_stream()

        assert result is not None
        best_client, score = result
        assert best_client in clients
        assert 0.0 <= score <= 100.0

    def test_get_all_stream_scores(self):
        """Test getting scores for all streams"""
        # Register multiple clients
        clients = ["client_1", "client_2"]
        for client in clients:
            self.scorer.register_client(client)
            # Update with some audio
            audio_data = np.random.normal(0, 0.1, 1000)
            self.scorer.update_stream_quality(client, audio_data)

        scores = self.scorer.get_all_stream_scores()

        assert len(scores) == 2
        for client in clients:
            assert client in scores
            assert 0.0 <= scores[client] <= 100.0

    def test_set_phone_distance(self):
        """Test setting phone distance"""
        client_id = "test_client_1"
        self.scorer.register_client(client_id)

        distance = 5.0
        success = self.scorer.set_phone_distance(client_id, distance)
        client_info = self.scorer.get_client_info(client_id)

        assert success is True
        assert client_info is not None
        assert client_info.quality_metrics.phone_distance == distance

    def test_set_phone_distance_nonexistent_client(self):
        """Test setting phone distance for non-existent client"""
        success = self.scorer.set_phone_distance("nonexistent_client", 5.0)
        assert success is False

    def test_get_client_info(self):
        """Test getting client info"""
        client_id = "test_client_1"
        device_type = "phone"
        location = {"lat": 37.7749, "lng": -122.4194}

        self.scorer.register_client(client_id, device_type=device_type, location=location)

        client_info = self.scorer.get_client_info(client_id)

        assert client_info is not None
        assert client_info.client_id == client_id
        assert client_info.device_type == device_type
        assert client_info.location == location
        assert client_info.is_active is True

    def test_get_client_info_nonexistent(self):
        """Test getting info for non-existent client"""
        client_info = self.scorer.get_client_info("nonexistent_client")
        assert client_info is None

    def test_cleanup_inactive_clients(self):
        """Test cleanup of inactive clients"""
        # Register clients
        clients = ["client_1", "client_2"]
        for client in clients:
            self.scorer.register_client(client)

        # Manually set one client as old (simulate timeout)
        old_time = datetime(2020, 1, 1, tzinfo=timezone.utc)
        self.scorer.clients["client_1"].last_update = old_time

        # Cleanup with short timeout
        removed = self.scorer.cleanup_inactive_clients(timeout_seconds=1)

        assert "client_1" in removed
        assert "client_1" not in self.scorer.clients
        assert "client_2" in self.scorer.clients


class TestStreamQualityMetrics:
    """Test suite for StreamQualityMetrics class"""

    def test_default_initialization(self):
        """Test default initialization of metrics"""
        metrics = StreamQualityMetrics()

        assert metrics.snr == 0.0
        assert metrics.speech_clarity == 0.0
        assert metrics.volume_level == 0.0
        assert metrics.noise_level == 0.0
        assert metrics.phone_distance is None
        assert isinstance(metrics.timestamp, datetime)
        assert metrics.sample_count == 0

    def test_custom_initialization(self):
        """Test custom initialization of metrics"""
        custom_time = datetime(2023, 1, 1, tzinfo=timezone.utc)

        metrics = StreamQualityMetrics(
            snr=25.0,
            speech_clarity=80.0,
            volume_level=0.15,
            noise_level=0.02,
            phone_distance=3.5,
            timestamp=custom_time,
            sample_count=10,
        )

        assert metrics.snr == 25.0
        assert metrics.speech_clarity == 80.0
        assert metrics.volume_level == 0.15
        assert metrics.noise_level == 0.02
        assert metrics.phone_distance == 3.5
        assert metrics.timestamp == custom_time
        assert metrics.sample_count == 10


class TestClientStreamInfo:
    """Test suite for ClientStreamInfo class"""

    def test_default_initialization(self):
        """Test default initialization of client info"""
        client_id = "test_client"
        info = ClientStreamInfo(client_id=client_id)

        assert info.client_id == client_id
        assert info.is_active is True
        assert isinstance(info.last_update, datetime)
        assert isinstance(info.quality_metrics, StreamQualityMetrics)
        assert info.device_type is None
        assert info.location is None

    def test_custom_initialization(self):
        """Test custom initialization of client info"""
        client_id = "test_client"
        device_type = "tablet"
        location = {"lat": 40.7128, "lng": -74.0060}
        custom_time = datetime(2023, 1, 1, tzinfo=timezone.utc)
        custom_metrics = StreamQualityMetrics(snr=20.0)

        info = ClientStreamInfo(
            client_id=client_id,
            is_active=False,
            last_update=custom_time,
            quality_metrics=custom_metrics,
            device_type=device_type,
            location=location,
        )

        assert info.client_id == client_id
        assert info.is_active is False
        assert info.last_update == custom_time
        assert info.quality_metrics == custom_metrics
        assert info.device_type == device_type
        assert info.location == location


    def test_set_phone_location(self):
        """Test setting phone location for a client"""
        scorer = AudioStreamScorer()
        scorer.register_client("test_client")
        
        location = {"latitude": 37.7749, "longitude": -122.4194, "accuracy": 5.0}
        result = scorer.set_phone_location("test_client", location)
        assert result is True
        
        client_info = scorer.get_client_info("test_client")
        assert client_info.quality_metrics.location == location

    def test_set_phone_location_nonexistent_client(self):
        """Test setting phone location for non-existent client"""
        scorer = AudioStreamScorer()
        location = {"latitude": 37.7749, "longitude": -122.4194}
        result = scorer.set_phone_location("nonexistent_client", location)
        assert result is False

    def test_set_phone_rssi(self):
        """Test setting phone RSSI for a client"""
        scorer = AudioStreamScorer()
        scorer.register_client("test_client")
        
        rssi = -50.0
        result = scorer.set_phone_rssi("test_client", rssi)
        assert result is True
        
        client_info = scorer.get_client_info("test_client")
        assert client_info.quality_metrics.rssi == rssi

    def test_set_phone_rssi_nonexistent_client(self):
        """Test setting phone RSSI for non-existent client"""
        scorer = AudioStreamScorer()
        rssi = -60.0
        result = scorer.set_phone_rssi("nonexistent_client", rssi)
        assert result is False

    def test_calculate_overall_score_with_location_and_rssi(self):
        """Test overall score calculation with location and RSSI"""
        scorer = AudioStreamScorer()
        
        # Test with good location and RSSI
        metrics = StreamQualityMetrics(
            snr=25.0,
            speech_clarity=80.0,
            volume_level=0.5,
            location={"accuracy": 5.0},  # Good accuracy
            rssi=-40.0  # Good RSSI
        )
        score = scorer.calculate_overall_score(metrics)
        assert 0 <= score <= 100
        
        # Test with poor location and RSSI
        poor_metrics = StreamQualityMetrics(
            snr=25.0,
            speech_clarity=80.0,
            volume_level=0.5,
            location={"accuracy": 100.0},  # Poor accuracy
            rssi=-85.0  # Poor RSSI
        )
        poor_score = scorer.calculate_overall_score(poor_metrics)
        assert poor_score < score  # Should be worse than good metrics

    def test_metrics_with_new_fields(self):
        """Test that new fields are preserved during metric updates"""
        scorer = AudioStreamScorer()
        scorer.register_client("test_client")
        
        # Set initial location and RSSI
        location = {"latitude": 40.7128, "longitude": -74.0060}
        rssi = -55.0
        scorer.set_phone_location("test_client", location)
        scorer.set_phone_rssi("test_client", rssi)
        
        # Update stream quality (should preserve location and RSSI)
        test_audio = np.random.normal(0, 0.1, 1000)
        scorer.update_stream_quality("test_client", test_audio)
        
        # Check that location and RSSI are preserved
        client_info = scorer.get_client_info("test_client")
        assert client_info.quality_metrics.location == location
        assert client_info.quality_metrics.rssi == rssi


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
