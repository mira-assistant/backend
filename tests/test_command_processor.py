"""
Unit tests for WakeWordDetector logic from command_processor.py
"""

import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import pytest

from command_processor import WakeWordDetector


class TestWakeWordDetector:
    def setup_method(self):
        self.detector = WakeWordDetector()

    def test_default_wake_words(self):
        wake_words = self.detector.get_wake_words()
        assert "hey mira" in wake_words
        assert len(wake_words) > 0

    def test_add_wake_word(self):
        result = self.detector.add_wake_word("hello assistant", sensitivity=0.8)
        assert result is True
        wake_words = self.detector.get_wake_words()
        assert "hello assistant" in wake_words
        assert wake_words["hello assistant"].sensitivity == 0.8

    def test_add_wake_word_full_config(self):
        result = self.detector.add_wake_word("custom trigger", sensitivity=0.9, min_confidence=0.7)
        assert result is True
        config = self.detector.get_wake_words()["custom trigger"]
        assert config.sensitivity == 0.9
        assert config.min_confidence == 0.7

    def test_add_empty_wake_word(self):
        result = self.detector.add_wake_word("")
        assert result is False

    def test_remove_wake_word(self):
        self.detector.add_wake_word("test removal")
        result = self.detector.remove_wake_word("test removal")
        assert result is True
        assert "test removal" not in self.detector.get_wake_words()

    def test_remove_nonexistent_wake_word(self):
        result = self.detector.remove_wake_word("nonexistent word")
        assert result is False

    def test_enable_disable_wake_word(self):
        self.detector.add_wake_word("test enable")
        assert self.detector.set_wake_word_enabled("test enable", True) is True
        assert self.detector.get_wake_words()["test enable"].enabled is True
        assert self.detector.set_wake_word_enabled("test enable", False) is True
        assert self.detector.get_wake_words()["test enable"].enabled is False

    def test_enable_disable_nonexistent(self):
        assert self.detector.set_wake_word_enabled("nonexistent", True) is False
        assert self.detector.set_wake_word_enabled("nonexistent", False) is False

    def test_process_text_with_wake_word(self):
        detection = self.detector.process_audio_text(
            "test_client", "Hey Mira, how are you today?", audio_length=2.5
        )
        assert detection is not None
        assert detection.wake_word == "hey mira"
        assert detection.client_id == "test_client"

    def test_process_text_without_wake_word(self):
        detection = self.detector.process_audio_text(
            "test_client", "Hello there, nice weather today", audio_length=1.5
        )
        assert detection is None

    def test_process_empty_text(self):
        detection = self.detector.process_audio_text("test_client", "")
        assert detection is None

    def test_wake_word_workflow(self):
        initial_count = len(self.detector.get_wake_words())
        self.detector.add_wake_word("workflow test word", sensitivity=0.8, min_confidence=0.6)
        wake_words = self.detector.get_wake_words()
        assert len(wake_words) == initial_count + 1
        assert "workflow test word" in wake_words
        # Detect
        detection = self.detector.process_audio_text(
            "workflow_client", "Please workflow test word and start listening"
        )
        assert detection is not None
        assert detection.wake_word == "workflow test word"
        # Disable
        self.detector.set_wake_word_enabled("workflow test word", False)
        detection2 = self.detector.process_audio_text(
            "workflow_client", "Please workflow test word and start listening"
        )
        assert detection2 is None
        # Remove
        self.detector.remove_wake_word("workflow test word")
        assert "workflow test word" not in self.detector.get_wake_words()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
