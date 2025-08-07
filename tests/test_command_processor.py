"""
Unit tests for command processing logic from command_processor.py
"""

import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import pytest
from unittest.mock import Mock, patch
from datetime import datetime, timezone

from processors.command_processor import (
    WakeWordDetector,
    WakeWordDetection,
    WakeWordConfig,
    CommandProcessor,
)
from models import Interaction


class TestWakeWordDetector:
    def setup_method(self):
        self.detector = WakeWordDetector()

    def test_initialization(self):
        """Test that detector initializes properly"""
        assert self.detector.sample_rate == 16000
        assert len(self.detector.wake_words) == 0
        assert len(self.detector.detection_callbacks) == 0

    def test_add_wake_word(self):
        """Test adding a wake word"""
        result = self.detector.add_wake_word("hello assistant", sensitivity=0.8)
        assert result is True
        assert "hello assistant" in self.detector.wake_words
        config = self.detector.wake_words["hello assistant"]
        assert config.sensitivity == 0.8

    def test_add_wake_word_full_config(self):
        """Test adding wake word with full configuration"""
        def test_callback():
            return "callback executed"
        
        result = self.detector.add_wake_word(
            "custom trigger", 
            sensitivity=0.9, 
            min_confidence=0.7,
            callback=test_callback
        )
        assert result is True
        config = self.detector.wake_words["custom trigger"]
        assert config.sensitivity == 0.9
        assert config.min_confidence == 0.7
        assert config.callback == test_callback

    def test_add_empty_wake_word(self):
        """Test that empty wake words are rejected"""
        result = self.detector.add_wake_word("")
        assert result is False
        result = self.detector.add_wake_word("   ")
        assert result is False

    def test_detect_wake_words_text_with_wake_word(self):
        """Test wake word detection in text"""
        # First add a wake word
        self.detector.add_wake_word("hey mira")
        
        detection = self.detector.detect_wake_words_text(
            "test_client", "Hey Mira, how are you today?", audio_length=2.5
        )
        assert detection is not None
        assert detection.wake_word == "hey mira"
        assert detection.client_id == "test_client"
        assert detection.audio_snippet_length == 2.5

    def test_detect_wake_words_text_without_wake_word(self):
        """Test text without wake words"""
        self.detector.add_wake_word("hey mira")
        
        detection = self.detector.detect_wake_words_text(
            "test_client", "Hello there, nice weather today", audio_length=1.5
        )
        assert detection is None

    def test_detect_wake_words_text_empty(self):
        """Test empty text detection"""
        detection = self.detector.detect_wake_words_text("test_client", "")
        assert detection is None

class TestWakeWordConfig:
    def test_wake_word_config_creation(self):
        """Test WakeWordConfig dataclass creation"""
        def test_callback():
            return "test"
        
        config = WakeWordConfig(
            word="test word",
            sensitivity=0.8,
            min_confidence=0.6,
            callback=test_callback
        )
        
        assert config.word == "test word"
        assert config.sensitivity == 0.8
        assert config.min_confidence == 0.6
        assert config.callback == test_callback


class TestWakeWordDetection:
    def test_wake_word_detection_creation(self):
        """Test WakeWordDetection dataclass creation"""
        detection = WakeWordDetection(
            wake_word="test",
            confidence=0.9,
            client_id="client123",
            callback=True,
            audio_snippet_length=2.5
        )
        
        assert detection.wake_word == "test"
        assert detection.confidence == 0.9
        assert detection.client_id == "client123"
        assert detection.callback is True
        assert detection.audio_snippet_length == 2.5
        assert isinstance(detection.timestamp, datetime)


class TestCommandProcessor:
    @patch('ml_model_manager.get_available_models')
    @patch('builtins.open')
    @patch('json.load')
    def test_initialization(self, mock_json_load, mock_open, mock_get_models):
        """Test CommandProcessor initialization"""
        mock_open.return_value.__enter__.return_value.read.return_value = "system prompt"
        mock_json_load.return_value = {"type": "object"}
        mock_get_models.return_value = [{"id": "llama-2-7b-chat-hf-function-calling-v3", "state": "loaded"}]
        
        processor = CommandProcessor()
        
        assert processor.model_manager is not None

    @patch('ml_model_manager.get_available_models')
    @patch('builtins.open')
    @patch('json.load')
    def test_load_model_tools(self, mock_json_load, mock_open, mock_get_models):
        """Test that model tools are loaded"""
        mock_open.return_value.__enter__.return_value.read.return_value = "system prompt"
        mock_json_load.return_value = {"type": "object"}
        mock_get_models.return_value = [{"id": "llama-2-7b-chat-hf-function-calling-v3", "state": "loaded"}]
        
        processor = CommandProcessor()
        
        # Verify that the model manager was created and has tools
        assert processor.model_manager is not None
        # The tools are registered in load_model_tools, we can't easily test the count
        # without accessing private MLModelManager internals

    @patch('ml_model_manager.get_available_models')
    @patch('builtins.open')
    @patch('json.load')
    @patch('ml_model_manager.client.chat.completions.create')
    def test_process_command(self, mock_create, mock_json_load, mock_open, mock_get_models):
        """Test command processing"""
        mock_open.return_value.__enter__.return_value.read.return_value = "system prompt"
        mock_json_load.return_value = {"type": "object"}
        mock_get_models.return_value = [{"id": "llama-2-7b-chat-hf-function-calling-v3", "state": "loaded"}]
        
        # Mock the OpenAI response with JSON content
        mock_response = Mock()
        mock_response.choices = [Mock()]
        mock_response.choices[0].message.content = '{"action": "response", "data": "AI response"}'
        mock_create.return_value = mock_response
        
        processor = CommandProcessor()
        
        # Create a mock interaction
        interaction = Mock()
        interaction.text = "What time is it?"
        
        result = processor.process_command(interaction)
        
        assert result == {"action": "response", "data": "AI response"}


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
