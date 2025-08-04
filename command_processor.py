"""
Command Processing System

This module implements a complete command processing system that includes wake word detection
and AI-powered command processing workflow. It monitors audio streams for trigger words
and processes commands through AI model integration for callback determination.

Features:
- Real-time wake word detection across multiple audio streams
- AI-powered command processing with callback execution
- Callback function registry for available commands
- Configurable wake words and sensitivity settings
- Integration with audio stream scorer for best stream selection
- Thread-safe operation for concurrent audio processing
"""

import json
import logging
import threading
from typing import Dict, List, Optional, Callable
from dataclasses import dataclass, field
from datetime import datetime, timezone
import tzlocal
from ml_model_manager import MLModelManager
from models import Interaction

logger = logging.getLogger(__name__)


@dataclass
class WakeWordDetection:
    """Container for wake word detection result"""

    wake_word: str
    confidence: float
    client_id: str
    callback: bool
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    audio_snippet_length: float = 0.0


@dataclass
class WakeWordConfig:
    """Configuration for a wake word"""

    word: str
    sensitivity: float = 0.7
    min_confidence: float = 0.5
    callback: Optional[Callable] = None


class WakeWordDetector:
    """
    Wake Word Detection System for monitoring audio streams.

    This class provides real-time wake word detection with support for
    multiple wake words, configurable sensitivity, and integration with
    the audio stream scoring system.
    """

    def __init__(self, sample_rate: int = 16000):
        """
        Initialize the wake word detector.

        Args:
            sample_rate: Audio sample rate for processing
        """
        self.sample_rate = sample_rate
        self.wake_words: Dict[str, WakeWordConfig] = {}
        self.detection_callbacks: List[Callable[[WakeWordDetection], None]] = []
        self._lock = threading.Lock()

        logger.info("WakeWordDetector initialized")

    def add_wake_word(
        self,
        word: str,
        sensitivity: float = 0.7,
        min_confidence: float = 0.5,
        callback: Optional[Callable] = None,
    ) -> bool:
        """
        Add a new wake word to the detection system.

        Args:
            word: The wake word or phrase to detect
            sensitivity: Detection threshold (0.0-1.0)
            min_confidence: Minimum confidence required for detection
            cooldown_seconds: Time to wait before detecting the same word again

        Returns:
            bool: True if wake word was added successfully
        """
        with self._lock:
            word_normalized = word.lower().strip()

            if not word_normalized:
                logger.warning("Cannot add empty wake word")
                return False

            config = WakeWordConfig(
                word=word_normalized,
                sensitivity=max(0.0, min(1.0, sensitivity)),
                min_confidence=max(0.0, min(1.0, min_confidence)),
                callback=callback,
            )

            self.wake_words[word_normalized] = config
            logger.info(f"Added wake word: '{word_normalized}' with sensitivity {sensitivity}")
            return True

    def detect_wake_words_text(
        self, client_id: str, transcribed_text: str, audio_length: float = 0.0
    ) -> Optional[WakeWordDetection]:
        """
        Process transcribed text for wake word detection.

        This method analyzes transcribed text from audio streams to detect wake words.
        It's designed to work with existing speech-to-text systems.

        Args:
            client_id: ID of the client that provided the audio
            transcribed_text: The text transcribed from audio
            audio_length: Length of the original audio snippet in seconds

        Returns:
            WakeWordDetection: Detection result if wake word found, None otherwise
        """
        if not transcribed_text:
            return None

        text_normalized = ''.join(
            c for c in transcribed_text.lower().strip() if c.isalnum() or c.isspace()
        )

        with self._lock:
            for wake_word, config in self.wake_words.items():

                confidence = self._calculate_text_confidence(wake_word, text_normalized)

                # Incorporate sensitivity: require higher confidence for lower sensitivity
                # Effective threshold = min_confidence + (1 - sensitivity) * (1 - min_confidence)
                effective_threshold = config.min_confidence + (1.0 - config.sensitivity) * (
                    1.0 - config.min_confidence
                )
                if confidence >= effective_threshold:
                    detection = WakeWordDetection(
                        wake_word=wake_word,
                        confidence=confidence,
                        client_id=client_id,
                        audio_snippet_length=audio_length,
                        callback=True if config.callback is not None else False,
                    )

                    if config.callback is not None:
                        config.callback()

                    return detection

        return None

    def _calculate_text_confidence(self, wake_word: str, text: str) -> float:
        """
        Calculate confidence score for wake word detection in text.

        Args:
            wake_word: The wake word to search for
            text: The text to search in

        Returns:
            float: Confidence score (0.0-1.0)
        """
        # Exact match
        if wake_word in text:
            return 1.0

        # Word-by-word matching
        wake_words = wake_word.split()
        text_words = text.split()

        if len(wake_words) == 1:
            # Single word - check for substring or similar
            for word in text_words:
                if wake_word in word or word in wake_word:
                    return 0.8
                # Simple fuzzy matching based on length and common characters
                if len(word) >= 3 and len(wake_word) >= 3:
                    common_chars = len(set(wake_word) & set(word))
                    similarity = common_chars / max(len(wake_word), len(word))
                    if similarity > 0.6:
                        return 0.6
        else:
            # Multi-word phrase - check for partial matches
            matches = 0
            for wake_word_part in wake_words:
                for text_word in text_words:
                    if wake_word_part in text_word or text_word in wake_word_part:
                        matches += 1
                        break

            if matches > 0:
                confidence = matches / len(wake_words)
                return min(0.9, confidence)

        return 0.0

    def _trigger_callbacks(self, detection: WakeWordDetection):
        """
        Trigger all registered detection callbacks.

        Args:
            detection: The wake word detection to pass to callbacks
        """
        for callback in self.detection_callbacks:
            try:
                callback(detection)
            except Exception as e:
                logger.error(f"Error in wake word detection callback: {e}")


class CommandProcessor:
    """Main command processing workflow orchestrator"""

    def __init__(self):
        """
        Initialize command processor

        Args:
            callback_registry: Optional callback registry, creates default if None
        """

        system_prompt = open("schemas/command_processing/system_prompt.txt", "r").read().strip()
        structured_response = json.load(open("schemas/command_processing/structured_output.json", "r"))
        self.model_manager = MLModelManager(
            model_name="llama-2-7b-chat",
            system_prompt=system_prompt,
            # structured_response=structured_response,
        )

        self.load_model_tools()

        logger.info("CommandProcessor initialized")

    def load_model_tools(self):
        """
        Load model tools for the AI model manager.
        This method can be extended to load additional tools as needed.
        """

        def get_weather(self, location: str = "current location") -> str:
            """Get weather information (placeholder implementation)"""
            # This is a placeholder implementation
            # In a real system, this would integrate with a weather API
            return f"The weather in {location} is partly cloudy with a temperature of 72°F"

        def get_time(self) -> str:
            """Get current time in user's timezone (auto-detected)"""
            local_tz = tzlocal.get_localzone()
            current_time = datetime.now(local_tz).strftime("%I:%M %p %Z")
            return f"The current time is {current_time}"

        def disable_mira(self) -> str:
            """Disable the Mira assistant service"""
            # Import here to avoid circular imports
            from mira import status

            status["enabled"] = False
            return "Mira assistant has been disabled. Say 'Hey Mira' to re-enable."

        self.model_manager.register_tool(get_weather, "Fetch Weather Information")
        self.model_manager.register_tool(get_time, "Fetch Current Time")
        self.model_manager.register_tool(disable_mira, "Disable Mira Assistant")

    def process_command(self, interaction: Interaction):
        """
        Process a command through the AI model and execute callbacks

        Args:
            interaction: The user interaction object
            client_id: ID of the client that triggered the command

        Returns:
            Result of command processing
        """

        logger.info(f"Processing command: {interaction.text}")
        response = self.model_manager.run_inference(
            interaction,
        )

        return response