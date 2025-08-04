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

import logging
import threading
import inspect
from typing import Dict, List, Optional, Callable, Any, Tuple
from dataclasses import dataclass, field
from datetime import datetime, timezone

logger = logging.getLogger(__name__)


@dataclass
class WakeWordDetection:
    """Container for wake word detection result"""

    wake_word: str
    confidence: float
    client_id: str
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    audio_snippet_length: float = 0.0  # Length of audio snippet in seconds


@dataclass
class WakeWordConfig:
    """Configuration for a wake word"""

    word: str
    sensitivity: float = 0.7
    min_confidence: float = 0.5


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

        # Default wake words
        self._setup_default_wake_words()

        logger.info("WakeWordDetector initialized")

    def _setup_default_wake_words(self):
        """Setup default wake words for the system"""
        default_words = ["hey mira", "okay mira", "mira", "listen mira", "start recording"]

        for word in default_words:
            self.add_wake_word(word, sensitivity=0.7)

    def add_wake_word(
        self,
        word: str,
        sensitivity: float = 0.7,
        min_confidence: float = 0.5,
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
            )

            self.wake_words[word_normalized] = config
            logger.info(f"Added wake word: '{word_normalized}' with sensitivity {sensitivity}")
            return True

    def remove_wake_word(self, word: str) -> bool:
        """
        Remove a wake word from the detection system.

        Args:
            word: The wake word to remove

        Returns:
            bool: True if wake word was removed successfully
        """
        with self._lock:
            word_normalized = word.lower().strip()

            if word_normalized in self.wake_words:
                del self.wake_words[word_normalized]
                logger.info(f"Removed wake word: '{word_normalized}'")
                return True

            logger.warning(f"Wake word '{word_normalized}' not found")
            return False

    def get_wake_words(self) -> Dict[str, WakeWordConfig]:
        """
        Get all configured wake words.

        Returns:
            Dict[str, WakeWordConfig]: Dictionary of wake word configurations
        """
        with self._lock:
            return self.wake_words.copy()

    def add_detection_callback(self, callback: Callable[[WakeWordDetection], None]):
        """
        Add a callback function to be called when a wake word is detected.

        Args:
            callback: Function to call with WakeWordDetection when detected
        """
        if callback not in self.detection_callbacks:
            self.detection_callbacks.append(callback)
            logger.info("Added wake word detection callback")

    def remove_detection_callback(self, callback: Callable[[WakeWordDetection], None]):
        """
        Remove a detection callback.

        Args:
            callback: Callback function to remove
        """
        if callback in self.detection_callbacks:
            self.detection_callbacks.remove(callback)
            logger.info("Removed wake word detection callback")

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

        text_normalized = transcribed_text.lower().strip()

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
                    )

                    self._trigger_callbacks(detection)

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


@dataclass
class CallbackFunction:
    """Represents a registered callback function"""

    name: str
    function: Callable
    description: str
    parameters: Dict[str, Any] = field(default_factory=dict)
    enabled: bool = True


@dataclass
class CommandProcessingResult:
    """Result of command processing workflow"""

    callback_executed: bool
    callback_name: Optional[str] = None
    callback_result: Any = None
    user_response: str = ""
    error: Optional[str] = None
    ai_response: Optional[Dict] = None


class CallbackRegistry:
    """Registry for managing available callback functions"""

    def __init__(self):
        self.callbacks: Dict[str, CallbackFunction] = {}
        self._setup_default_callbacks()
        logger.info("CallbackRegistry initialized")

    def _setup_default_callbacks(self):
        """Setup default callback functions"""
        # Register core callback functions
        self.register(
            "getWeather", self._get_weather, "Get current weather information for a location"
        )
        self.register("getTime", self._get_time, "Get current time")
        self.register("disableMira", self._disable_mira, "Disable the Mira assistant service")

    def register(
        self,
        name: str,
        function: Callable,
        description: str,
        parameters: Optional[Dict[str, Any]] = None,
    ) -> bool:
        """
        Register a new callback function

        Args:
            name: Function name to register
            function: The callable function
            description: Description of what the function does
            parameters: Optional parameter schema

        Returns:
            bool: True if registration successful
        """
        if not callable(function):
            logger.error(f"Cannot register non-callable object as callback: {name}")
            return False

        # Auto-extract parameters from function signature
        if parameters is None:
            sig = inspect.signature(function)
            parameters = {}
            for param_name, param in sig.parameters.items():
                param_info = {
                    "type": (
                        param.annotation.__name__
                        if param.annotation != inspect.Parameter.empty
                        else "str"
                    ),
                    "required": param.default == inspect.Parameter.empty,
                }
                parameters[param_name] = param_info

        callback = CallbackFunction(
            name=name, function=function, description=description, parameters=parameters
        )

        self.callbacks[name] = callback
        logger.info(f"Registered callback function: {name}")
        return True

    def unregister(self, name: str) -> bool:
        """
        Unregister a callback function

        Args:
            name: Function name to unregister

        Returns:
            bool: True if unregistration successful
        """
        if name in self.callbacks:
            del self.callbacks[name]
            logger.info(f"Unregistered callback function: {name}")
            return True

        logger.warning(f"Callback function not found: {name}")
        return False

    def execute(self, name: str, **kwargs) -> Tuple[bool, Any]:
        """
        Execute a registered callback function

        Args:
            name: Function name to execute
            **kwargs: Arguments to pass to the function

        Returns:
            Tuple[bool, Any]: (success, result)
        """
        if name not in self.callbacks:
            logger.error(f"Callback function not found: {name}")
            return False, f"Function '{name}' not found"

        callback = self.callbacks[name]
        if not callback.enabled:
            logger.warning(f"Callback function disabled: {name}")
            return False, f"Function '{name}' is disabled"

        result = callback.function(**kwargs)
        logger.info(f"Successfully executed callback: {name}")
        return True, result

    def get_function_list(self) -> List[str]:
        """Get list of available function names"""
        return [name for name, callback in self.callbacks.items() if callback.enabled]

    def get_function_descriptions(self) -> Dict[str, str]:
        """Get mapping of function names to descriptions"""
        return {
            name: callback.description
            for name, callback in self.callbacks.items()
            if callback.enabled
        }

    def set_enabled(self, name: str, enabled: bool) -> bool:
        """Enable or disable a callback function"""
        if name in self.callbacks:
            self.callbacks[name].enabled = enabled
            status = "enabled" if enabled else "disabled"
            logger.info(f"Callback function {name} {status}")
            return True
        return False

    # Default callback implementations
    def _get_weather(self, location: str = "current location") -> str:
        """Get weather information (placeholder implementation)"""
        # This is a placeholder implementation
        # In a real system, this would integrate with a weather API
        return f"The weather in {location} is partly cloudy with a temperature of 72°F"

    def _get_time(self) -> str:
        """Get current time"""
        current_time = datetime.now().strftime("%I:%M %p")
        return f"The current time is {current_time}"

    def _disable_mira(self) -> str:
        """Disable the Mira assistant service"""
        # Import here to avoid circular imports
        from mira import status

        status["enabled"] = False
        return "Mira assistant has been disabled. Say 'Hey Mira' to re-enable."


class CommandProcessor:
    """Main command processing workflow orchestrator"""

    def __init__(self, callback_registry: Optional[CallbackRegistry] = None):
        """
        Initialize command processor

        Args:
            callback_registry: Optional callback registry, creates default if None
        """
        self.callback_registry = callback_registry or CallbackRegistry()
        # Initialize ML model manager for command inference
        from ml_model_manager import MLModelManager
        # Use a lightweight NLP model suitable for command processing
        # Note: Model validation will happen when MLModelManager is instantiated
        self.ml_model_manager = MLModelManager(model_name="microsoft/DialoGPT-small")
        logger.info("CommandProcessor initialized")

    def process_command(
        self, interaction_text: str, client_id: str
    ) -> CommandProcessingResult:
        """
        Process a command through the AI model and execute callbacks

        Args:
            interaction_text: The transcribed user interaction
            client_id: ID of the client that triggered the command

        Returns:
            CommandProcessingResult: Result of command processing
        """

        logger.info(f"Processing command from client {client_id}: {interaction_text}")

        # Get AI response for callback determination
        ai_response = self._get_ai_callback_response(interaction_text)

        if not ai_response:
            return CommandProcessingResult(
                callback_executed=False,
                user_response="I didn't understand that command. Could you please try again?",
                error="No AI response received",
            )

        # Extract callback information from AI response
        callback_name = ai_response.get("callback_function")
        callback_args = ai_response.get("callback_arguments", {})
        user_response = ai_response.get("user_response", "")

        # Execute callback if specified
        if callback_name:
            success, result = self.callback_registry.execute(callback_name, **callback_args)

            if success:
                # Enhance user response with callback result if needed
                if result and isinstance(result, str):
                    user_response = result if not user_response else f"{user_response} {result}"

                return CommandProcessingResult(
                    callback_executed=True,
                    callback_name=callback_name,
                    callback_result=result,
                    user_response=user_response,
                    ai_response=ai_response,
                )
            else:
                return CommandProcessingResult(
                    callback_executed=False,
                    user_response=f"Sorry, I couldn't execute that command: {result}",
                    error=str(result),
                    ai_response=ai_response,
                )
        else:
            # No callback needed, just return AI response
            return CommandProcessingResult(
                callback_executed=False,
                user_response=user_response or "I understand, but no action is needed right now.",
                ai_response=ai_response,
            )

    def process_command_with_recursion(
        self, interaction_text: str, client_id: str, max_recursion: int = 3
    ) -> CommandProcessingResult:
        """
        Process a command with recursive callback execution support

        Args:
            interaction_text: The transcribed user interaction
            client_id: ID of the client that triggered the command
            max_recursion: Maximum recursion depth for callback chains

        Returns:
            CommandProcessingResult: Result of command processing with recursive execution
        """
        logger.info(f"Processing recursive command from client {client_id}: {interaction_text}")

        # Get available functions from callback registry
        function_list = self.callback_registry.get_function_list()
        function_descriptions = self.callback_registry.get_function_descriptions()

        # Define callback executor function
        def callback_executor(name: str, args: Dict[str, Any]) -> Any:
            success, result = self.callback_registry.execute(name, **args)
            if not success:
                raise Exception(f"Callback execution failed: {result}")
            return result

        # Use ML model manager for recursive command processing
        response = self.ml_model_manager.process_recursive_command(
            interaction_text,
            function_list,
            function_descriptions,
            callback_executor,
            max_recursion
        )

        if not response.success:
            return CommandProcessingResult(
                callback_executed=False,
                user_response=response.user_response,
                error=response.error,
            )

        # Convert ModelResponse to CommandProcessingResult
        return CommandProcessingResult(
            callback_executed=response.callback_function is not None,
            callback_name=response.callback_function,
            callback_result=None,  # Result is embedded in the recursive processing
            user_response=response.user_response,
            ai_response={
                "callback_function": response.callback_function,
                "callback_arguments": response.callback_arguments,
                "user_response": response.user_response
            },
        )

    def _get_ai_callback_response(
        self, interaction_text: str
    ) -> Optional[Dict]:
        """
        Get AI model response for callback determination

        Args:
            interaction_text: User interaction text

        Returns:
            Dict: AI response with callback information
        """
        # Get available functions from callback registry
        function_list = self.callback_registry.get_function_list()
        function_descriptions = self.callback_registry.get_function_descriptions()

        # Use ML model manager for command inference - let exceptions propagate
        response = self.ml_model_manager.process_command_inference(
            interaction_text,
            function_list,
            function_descriptions
        )

        # Convert ModelResponse to expected dict format
        return {
            "callback_function": response.callback_function,
            "callback_arguments": response.callback_arguments,
            "user_response": response.user_response
        }





