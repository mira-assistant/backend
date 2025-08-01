"""
Tests for the Wake Word Detection System
"""

import pytest
from datetime import datetime, timezone
from unittest.mock import Mock

# Handle missing dependencies gracefully
try:
    from wake_word_detector import WakeWordDetector, WakeWordConfig, WakeWordDetection
    WAKE_WORD_AVAILABLE = True
except ImportError as e:
    print(f"Skipping wake word detection tests due to missing dependencies: {e}")
    WAKE_WORD_AVAILABLE = False
    
    # Create dummy classes for type hints
    class WakeWordDetector:
        pass
    class WakeWordConfig:
        pass
    class WakeWordDetection:
        pass


@pytest.mark.skipif(not WAKE_WORD_AVAILABLE, reason="Wake word detection dependencies not available")
class TestWakeWordDetector:
    """Test suite for WakeWordDetector class"""
    
    def setup_method(self):
        """Set up test fixtures"""
        self.detector = WakeWordDetector(sample_rate=16000)
    
    def test_initialization(self):
        """Test detector initialization"""
        assert self.detector.sample_rate == 16000
        assert self.detector.enabled is True
        assert len(self.detector.wake_words) > 0  # Should have default wake words
        assert "hey mira" in self.detector.wake_words
        assert "mira" in self.detector.wake_words
    
    def test_add_wake_word(self):
        """Test adding a new wake word"""
        word = "hello assistant"
        success = self.detector.add_wake_word(word, sensitivity=0.8)
        
        assert success is True
        assert word in self.detector.wake_words
        assert self.detector.wake_words[word].sensitivity == 0.8
        assert self.detector.wake_words[word].enabled is True
    
    def test_add_empty_wake_word(self):
        """Test adding an empty wake word"""
        success = self.detector.add_wake_word("", sensitivity=0.8)
        assert success is False
        
        success = self.detector.add_wake_word("   ", sensitivity=0.8)
        assert success is False
    
    def test_remove_wake_word(self):
        """Test removing a wake word"""
        word = "test word"
        self.detector.add_wake_word(word)
        assert word in self.detector.wake_words
        
        success = self.detector.remove_wake_word(word)
        assert success is True
        assert word not in self.detector.wake_words
    
    def test_remove_nonexistent_wake_word(self):
        """Test removing a wake word that doesn't exist"""
        success = self.detector.remove_wake_word("nonexistent word")
        assert success is False
    
    def test_enable_disable_wake_word(self):
        """Test enabling and disabling a wake word"""
        word = "test word"
        self.detector.add_wake_word(word)
        
        # Disable
        success = self.detector.set_wake_word_enabled(word, False)
        assert success is True
        assert self.detector.wake_words[word].enabled is False
        
        # Enable
        success = self.detector.set_wake_word_enabled(word, True)
        assert success is True
        assert self.detector.wake_words[word].enabled is True
    
    def test_enable_disable_nonexistent_wake_word(self):
        """Test enabling/disabling a wake word that doesn't exist"""
        success = self.detector.set_wake_word_enabled("nonexistent word", True)
        assert success is False
    
    def test_get_wake_words(self):
        """Test getting all wake words"""
        wake_words = self.detector.get_wake_words()
        
        assert isinstance(wake_words, dict)
        assert len(wake_words) > 0
        assert "hey mira" in wake_words
        assert isinstance(wake_words["hey mira"], WakeWordConfig)
    
    def test_process_audio_text_exact_match(self):
        """Test wake word detection with exact text match"""
        client_id = "test_client"
        text = "Hey Mira, what's the weather today?"
        
        detection = self.detector.process_audio_text(client_id, text, audio_length=2.5)
        
        assert detection is not None
        assert detection.wake_word == "hey mira"
        assert detection.confidence == 1.0
        assert detection.client_id == client_id
        assert detection.audio_snippet_length == 2.5
    
    def test_process_audio_text_case_insensitive(self):
        """Test wake word detection is case insensitive"""
        client_id = "test_client"
        text = "HEY MIRA, turn on the lights"
        
        detection = self.detector.process_audio_text(client_id, text)
        
        assert detection is not None
        assert detection.wake_word == "hey mira"
        assert detection.confidence == 1.0
    
    def test_process_audio_text_partial_match(self):
        """Test wake word detection with partial matches"""
        client_id = "test_client"
        text = "Hey Meera, can you help me?"  # Similar to "Hey Mira"
        
        detection = self.detector.process_audio_text(client_id, text)
        
        # Should detect "mira" with lower confidence due to similarity
        if detection:
            assert detection.confidence < 1.0
            assert detection.confidence > 0.0
    
    def test_process_audio_text_single_word_match(self):
        """Test wake word detection with single word"""
        client_id = "test_client"
        text = "Mira, are you listening?"
        
        detection = self.detector.process_audio_text(client_id, text)
        
        assert detection is not None
        # Could detect either "mira" or "hey mira" depending on algorithm order
        assert detection.wake_word in ["mira", "hey mira"]
        assert detection.confidence >= 0.5  # Should meet minimum confidence
    
    def test_process_audio_text_no_match(self):
        """Test wake word detection with no matching text"""
        client_id = "test_client"
        text = "Hello there, how are you doing today?"
        
        detection = self.detector.process_audio_text(client_id, text)
        
        assert detection is None
    
    def test_process_audio_text_empty_text(self):
        """Test wake word detection with empty text"""
        client_id = "test_client"
        text = ""
        
        detection = self.detector.process_audio_text(client_id, text)
        
        assert detection is None
    
    def test_process_audio_text_disabled_detector(self):
        """Test wake word detection when detector is disabled"""
        self.detector.set_enabled(False)
        
        client_id = "test_client"
        text = "Hey Mira, hello"
        
        detection = self.detector.process_audio_text(client_id, text)
        
        assert detection is None
    
    def test_process_audio_text_disabled_wake_word(self):
        """Test wake word detection when specific wake word is disabled"""
        self.detector.set_wake_word_enabled("hey mira", False)
        
        client_id = "test_client"
        text = "Hey Mira, what's up?"
        
        detection = self.detector.process_audio_text(client_id, text)
        
        # Should detect a different wake word since "hey mira" is disabled
        if detection:
            assert detection.wake_word != "hey mira"
            # Could be "mira", "okay mira", etc.
            assert "mira" in detection.wake_word
    
    def test_cooldown_functionality(self):
        """Test wake word detection cooldown"""
        # Set a short cooldown
        word = "test word"
        self.detector.add_wake_word(word, cooldown_seconds=0.1)
        
        client_id = "test_client"
        text = f"{word} hello"
        
        # First detection should work
        detection1 = self.detector.process_audio_text(client_id, text)
        assert detection1 is not None
        
        # Immediate second detection should be blocked by cooldown
        detection2 = self.detector.process_audio_text(client_id, text)
        assert detection2 is None
        
        # After cooldown, should work again
        import time
        time.sleep(0.2)
        detection3 = self.detector.process_audio_text(client_id, text)
        assert detection3 is not None
    
    def test_detection_callbacks(self):
        """Test wake word detection callbacks"""
        callback_results = []
        
        def test_callback(detection):
            callback_results.append(detection)
        
        self.detector.add_detection_callback(test_callback)
        
        client_id = "test_client"
        text = "Hey Mira, test callback"
        
        detection = self.detector.process_audio_text(client_id, text)
        
        assert detection is not None
        assert len(callback_results) == 1
        assert callback_results[0] == detection
    
    def test_remove_detection_callback(self):
        """Test removing detection callbacks"""
        callback_results = []
        
        def test_callback(detection):
            callback_results.append(detection)
        
        self.detector.add_detection_callback(test_callback)
        self.detector.remove_detection_callback(test_callback)
        
        client_id = "test_client"
        text = "Hey Mira, test callback removal"
        
        detection = self.detector.process_audio_text(client_id, text)
        
        assert detection is not None
        assert len(callback_results) == 0  # Callback should not have been called
    
    def test_get_recent_detections(self):
        """Test getting recent detections"""
        client_id = "test_client"
        
        # Generate some detections
        texts = ["Hey Mira", "Mira help", "Hey Mira again"]
        for text in texts:
            self.detector.process_audio_text(client_id, text)
        
        detections = self.detector.get_recent_detections(limit=5)
        
        assert len(detections) >= 2  # Should have at least 2 detections (accounting for cooldown)
        assert all(isinstance(d, WakeWordDetection) for d in detections)
    
    def test_clear_detections(self):
        """Test clearing all detections"""
        client_id = "test_client"
        text = "Hey Mira, test clear"
        
        # Generate a detection
        self.detector.process_audio_text(client_id, text)
        
        # Verify detection exists
        detections_before = self.detector.get_recent_detections()
        assert len(detections_before) > 0
        
        # Clear detections
        self.detector.clear_detections()
        
        # Verify detections are cleared
        detections_after = self.detector.get_recent_detections()
        assert len(detections_after) == 0
    
    def test_get_stats(self):
        """Test getting detector statistics"""
        stats = self.detector.get_stats()
        
        assert isinstance(stats, dict)
        assert "enabled" in stats
        assert "total_wake_words" in stats
        assert "enabled_wake_words" in stats
        assert "total_detections" in stats
        assert "detections_by_wake_word" in stats
        assert "active_callbacks" in stats
        assert "numpy_available" in stats
        
        assert stats["enabled"] is True
        assert stats["total_wake_words"] > 0
        assert isinstance(stats["detections_by_wake_word"], dict)
    
    def test_process_audio_raw_placeholder(self):
        """Test raw audio processing (currently a placeholder)"""
        client_id = "test_client"
        audio_data = [1, 2, 3, 4, 5]  # Dummy audio data
        
        # Should return None since it's not implemented yet
        detection = self.detector.process_audio_raw(client_id, audio_data, audio_length=1.0)
        
        assert detection is None


@pytest.mark.skipif(not WAKE_WORD_AVAILABLE, reason="Wake word detection dependencies not available")
class TestWakeWordConfig:
    """Test suite for WakeWordConfig class"""
    
    def test_default_initialization(self):
        """Test default initialization of wake word config"""
        config = WakeWordConfig(word="test")
        
        assert config.word == "test"
        assert config.sensitivity == 0.7
        assert config.enabled is True
        assert config.min_confidence == 0.5
        assert config.cooldown_seconds == 2.0
    
    def test_custom_initialization(self):
        """Test custom initialization of wake word config"""
        config = WakeWordConfig(
            word="custom word",
            sensitivity=0.9,
            enabled=False,
            min_confidence=0.8,
            cooldown_seconds=5.0
        )
        
        assert config.word == "custom word"
        assert config.sensitivity == 0.9
        assert config.enabled is False
        assert config.min_confidence == 0.8
        assert config.cooldown_seconds == 5.0


@pytest.mark.skipif(not WAKE_WORD_AVAILABLE, reason="Wake word detection dependencies not available")
class TestWakeWordDetection:
    """Test suite for WakeWordDetection class"""
    
    def test_default_initialization(self):
        """Test default initialization of wake word detection"""
        detection = WakeWordDetection(
            wake_word="test word",
            confidence=0.95,
            client_id="test_client"
        )
        
        assert detection.wake_word == "test word"
        assert detection.confidence == 0.95
        assert detection.client_id == "test_client"
        assert isinstance(detection.timestamp, datetime)
        assert detection.audio_snippet_length == 0.0
    
    def test_custom_initialization(self):
        """Test custom initialization of wake word detection"""
        custom_time = datetime(2023, 1, 1, tzinfo=timezone.utc)
        
        detection = WakeWordDetection(
            wake_word="custom word",
            confidence=0.85,
            client_id="custom_client",
            timestamp=custom_time,
            audio_snippet_length=3.5
        )
        
        assert detection.wake_word == "custom word"
        assert detection.confidence == 0.85
        assert detection.client_id == "custom_client"
        assert detection.timestamp == custom_time
        assert detection.audio_snippet_length == 3.5


if __name__ == "__main__":
    pytest.main([__file__, "-v"])