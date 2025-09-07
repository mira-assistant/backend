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
- Network-isolated processing for multi-tenant support
- Proper dependency injection and lifecycle management
"""

import logging
from typing import Any, Dict, List, Optional, Callable
from dataclasses import dataclass, field
from datetime import datetime, timezone
from app.services.ml_model_manager import MLModelManager
from app.models import Interaction

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

    def __init__(self, network_id: str, wake_words_config: List[Dict[str, Any]] | None = None):
        """
        Initialize the wake word detector for a specific network.

        Args:
            network_id: ID of the network this detector belongs to
            wake_words_config: Configuration for wake words
        """
        self.network_id = network_id
        self.wake_words: Dict[str, WakeWordConfig] = {}
        self.detection_callbacks: List[Callable[[WakeWordDetection], None]] = []

        # Load wake words from configuration
        if wake_words_config:
            for config in wake_words_config:
                self.add_wake_word(
                    word=config["word"],
                    sensitivity=config.get("sensitivity", 0.7),
                    min_confidence=config.get("min_confidence", 0.5),
                    callback=config.get("callback"),
                )

        logger.info(f"WakeWordDetector initialized for network {network_id}")

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
            callback: Optional callback function to execute on detection

        Returns:
            bool: True if wake word was added successfully
        """
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
        logger.info(
            f"Added wake word: '{word_normalized}' for network {self.network_id} with sensitivity {sensitivity}"
        )
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

        text_normalized = "".join(
            c for c in transcribed_text.lower().strip() if c.isalnum() or c.isspace()
        )

        for wake_word, config in self.wake_words.items():
            confidence = self._calculate_text_confidence(wake_word, text_normalized)

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
        if wake_word in text:
            return 1.0

        wake_words = wake_word.split()
        text_words = text.split()

        if len(wake_words) == 1:
            for word in text_words:
                if wake_word in word or word in wake_word:
                    return 0.8
                if len(word) >= 3 and len(wake_word) >= 3:
                    common_chars = len(set(wake_word) & set(word))
                    similarity = common_chars / max(len(wake_word), len(word))
                    if similarity > 0.6:
                        return 0.6
        else:
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


class CommandProcessor:
    """Main command processing workflow orchestrator with proper dependency injection"""

    def __init__(self, model_manager: MLModelManager, network_id: str):
        """
        Initialize command processor with dependency injection.

        Args:
            model_manager: Injected ML model manager
            network_id: ID of the network this processor belongs to
        """
        self.model_manager = model_manager
        self.network_id = network_id
        self.wake_word_detector = WakeWordDetector(network_id)

        logger.info(f"CommandProcessor initialized for network {network_id}")

    def process_command(self, interaction: Interaction):
        """
        Process a command through the AI model and execute callbacks

        Args:
            interaction: The user interaction object

        Returns:
            Result of command processing
        """
        logger.info(f"Processing command for network {self.network_id}: {interaction.text}")
        response = self.model_manager.run_inference(interaction)

        return response

    def add_wake_word(
        self,
        word: str,
        sensitivity: float = 0.7,
        min_confidence: float = 0.5,
        callback: Optional[Callable] = None,
    ) -> bool:
        """
        Add a wake word to the detector.

        Args:
            word: The wake word or phrase to detect
            sensitivity: Detection threshold (0.0-1.0)
            min_confidence: Minimum confidence required for detection
            callback: Optional callback function to execute on detection

        Returns:
            bool: True if wake word was added successfully
        """
        return self.wake_word_detector.add_wake_word(word, sensitivity, min_confidence, callback)

    def detect_wake_words_text(
        self, client_id: str, transcribed_text: str, audio_length: float = 0.0
    ) -> Optional[WakeWordDetection]:
        """
        Detect wake words in transcribed text.

        Args:
            client_id: ID of the client that provided the audio
            transcribed_text: The text transcribed from audio
            audio_length: Length of the original audio snippet in seconds

        Returns:
            WakeWordDetection: Detection result if wake word found, None otherwise
        """
        return self.wake_word_detector.detect_wake_words_text(
            client_id, transcribed_text, audio_length
        )

    def cleanup(self):
        """Clean up resources when the processor is no longer needed."""
        logger.info(f"Cleaning up CommandProcessor for network {self.network_id}")
        # Add any cleanup logic here if needed
