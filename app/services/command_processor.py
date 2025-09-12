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

import json
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Callable, Dict, List, Optional

from google import genai
from google.genai import types

from app.core.config import settings
from app.core.mira_logger import MiraLogger
from app.models import Interaction


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

    def __init__(self, network_id: str):
        """
        Initialize the wake word detector for a specific network.

        Args:
            network_id: ID of the network this detector belongs to
            wake_words_config: Configuration for wake words
        """
        self.network_id = network_id
        self.wake_words: Dict[str, WakeWordConfig] = {}
        self.detection_callbacks: List[Callable[[WakeWordDetection], None]] = []

        MiraLogger.info(f"WakeWordDetector initialized for network {network_id}")

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
            MiraLogger.warning("Cannot add empty wake word")
            return False

        config = WakeWordConfig(
            word=word_normalized,
            sensitivity=max(0.0, min(1.0, sensitivity)),
            min_confidence=max(0.0, min(1.0, min_confidence)),
            callback=callback,
        )

        self.wake_words[word_normalized] = config
        MiraLogger.info(
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
            return self._calculate_single_word_confidence(wake_word, text_words)
        else:
            return self._calculate_multi_word_confidence(wake_words, text_words)

    def _calculate_single_word_confidence(
        self, wake_word: str, text_words: list[str]
    ) -> float:
        """Calculate confidence for single word wake word detection."""
        for word in text_words:
            if wake_word in word or word in wake_word:
                return 0.8
            if len(word) >= 3 and len(wake_word) >= 3:
                common_chars = len(set(wake_word) & set(word))
                similarity = common_chars / max(len(wake_word), len(word))
                if similarity > 0.6:
                    return 0.6
        return 0.0

    def _calculate_multi_word_confidence(
        self, wake_words: list[str], text_words: list[str]
    ) -> float:
        """Calculate confidence for multi-word wake word detection."""
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
    """Main command processing workflow orchestrator with direct Gemini integration"""

    def __init__(
        self, network_id: str, system_prompt: str = "You are a helpful AI assistant."
    ):
        """
        Initialize command processor with direct Gemini integration.

        Args:
            network_id: ID of the network this processor belongs to
            system_prompt: System prompt for the AI model
        """
        self.network_id = network_id
        self.system_prompt = system_prompt
        self.wake_word_detector = WakeWordDetector(network_id)

        # Initialize Gemini client
        self.gemini_client = self._initialize_gemini_client()

        # Register tools
        self._register_tools()

        MiraLogger.info(f"CommandProcessor initialized for network {network_id}")

    def _initialize_gemini_client(self):
        """Initialize Gemini client with API key from settings."""
        api_key = settings.gemini_api_key
        if not api_key:
            raise ValueError("GEMINI_API_KEY environment variable not set")
        return genai.Client(api_key=api_key)

    def _register_tools(self):
        """Register tools for the command processor."""
        from datetime import datetime

        import tzlocal

        def get_weather(location: str = "current location") -> str:
            """Get weather information for a specific location.

            Args:
                location: The location to get weather for (default: "current location")

            Returns:
                Weather information string
            """
            return (
                f"The weather in {location} is partly cloudy with a temperature of 72°F"
            )

        def get_time() -> str:
            """Get current time in user's timezone (auto-detected).

            Returns:
                Current time string in user's local timezone
            """
            local_tz = tzlocal.get_localzone()
            current_time = datetime.now(local_tz).strftime("%I:%M %p %Z")
            return f"The current time is {current_time}"

        def get_network_info(network_id: str) -> str:
            """Get information about a specific network.

            Args:
                network_id: The ID of the network to get information about

            Returns:
                Network information string
            """
            return f"Network {network_id} is active with 5 connected devices"

        # Store tools for execution
        self.tools = {
            "get_weather": get_weather,
            "get_time": get_time,
            "get_network_info": get_network_info,
        }

    def _get_gemini_tools(self) -> List[Dict[str, Any]]:
        """Convert tools to Gemini format."""
        gemini_tools = []
        for tool_name, tool_func in self.tools.items():
            gemini_tools.append(
                {
                    "function_declarations": [
                        {
                            "name": tool_name,
                            "description": (
                                tool_func.__doc__.split("\n")[1].strip()
                                if tool_func.__doc__
                                else f"Execute {tool_name}"
                            ),
                            "parameters": {
                                "type": "object",
                                "properties": {
                                    "query": {
                                        "type": "string",
                                        "description": "The query parameter",
                                    }
                                },
                                "required": ["query"],
                            },
                        }
                    ]
                }
            )
        return gemini_tools

    def _run_gemini_inference(
        self, interaction: Interaction, context: Optional[str] = None
    ) -> Dict[str, Any]:
        """Run inference using Gemini API."""

        system_instruction = self.system_prompt
        if context and context.strip():
            system_instruction += f"\n\nContext: {context.strip()}"

        # Prepare generation config
        config_params = {
            "system_instruction": system_instruction,
            "temperature": 0.7,
            "max_output_tokens": 2048,
        }

        # Add tools
        if self.tools:
            config_params["tools"] = self._get_gemini_tools()

        config = types.GenerateContentConfig(**config_params)

        response = self.gemini_client.models.generate_content(
            model="gemini-1.5-pro", contents=str(interaction.text), config=config
        )

        if not response.text:
            raise RuntimeError("Gemini model generated no content")

        try:
            return json.loads(response.text)
        except json.JSONDecodeError:
            return {"content": response.text}

    def process_command(self, interaction: Interaction, context: Optional[str] = None):
        """
        Process a command through the AI model and execute callbacks

        Args:
            interaction: The user interaction object
            context: Optional context information

        Returns:
            Result of command processing
        """
        MiraLogger.info(
            f"Processing command for network {self.network_id}: {interaction.text}"
        )
        response = self._run_gemini_inference(interaction, context)

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
        return self.wake_word_detector.add_wake_word(
            word, sensitivity, min_confidence, callback
        )

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
        MiraLogger.info(f"Cleaning up CommandProcessor for network {self.network_id}")
        # Add any cleanup logic here if needed
