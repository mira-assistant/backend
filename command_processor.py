"""
Wake Word Detection System

This module implements a wake word detection system that monitors audio streams
for specific trigger words and phrases. It integrates with the audio stream scoring
system to provide real-time wake word detection across multiple client streams.

Features:
- Real-time wake word detection across multiple audio streams
- Configurable wake words and sensitivity settings
- Integration with audio stream scorer for best stream selection
- Placeholder for advanced ML-based detection
- Thread-safe operation for concurrent audio processing
"""

import logging
import threading
import time
from typing import Dict, List, Optional, Callable, Any
from dataclasses import dataclass, field
from datetime import datetime, timezone
import re

import numpy as np

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
    sensitivity: float = 0.7  # Detection threshold (0.0-1.0)
    enabled: bool = True
    min_confidence: float = 0.5
    cooldown_seconds: float = 2.0  # Prevent rapid re-triggering


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
        self.recent_detections: List[WakeWordDetection] = []
        self.last_detection_time: Dict[str, float] = {}  # Per wake word cooldown
        self._lock = threading.Lock()
        self.enabled = True
        
        # Default wake words
        self._setup_default_wake_words()
        
        logger.info("WakeWordDetector initialized")
    
    def _setup_default_wake_words(self):
        """Setup default wake words for the system"""
        default_words = [
            "hey mira",
            "okay mira", 
            "mira",
            "listen mira",
            "start recording"
        ]
        
        for word in default_words:
            self.add_wake_word(word, sensitivity=0.7)
    
    def add_wake_word(self, word: str, sensitivity: float = 0.7, 
                      min_confidence: float = 0.5, cooldown_seconds: float = 2.0) -> bool:
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
                cooldown_seconds=max(0.1, cooldown_seconds)
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
    
    def set_wake_word_enabled(self, word: str, enabled: bool) -> bool:
        """
        Enable or disable a specific wake word.
        
        Args:
            word: The wake word to modify
            enabled: Whether the wake word should be enabled
            
        Returns:
            bool: True if operation was successful
        """
        with self._lock:
            word_normalized = word.lower().strip()
            
            if word_normalized in self.wake_words:
                self.wake_words[word_normalized].enabled = enabled
                status = "enabled" if enabled else "disabled"
                logger.info(f"Wake word '{word_normalized}' {status}")
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
    
    def process_audio_text(self, client_id: str, transcribed_text: str, 
                          audio_length: float = 0.0) -> Optional[WakeWordDetection]:
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
        if not self.enabled or not transcribed_text:
            return None
        
        text_normalized = transcribed_text.lower().strip()
        current_time = time.time()
        
        with self._lock:
            for wake_word, config in self.wake_words.items():
                if not config.enabled:
                    continue
                
                # Check cooldown
                last_detection = self.last_detection_time.get(f"{client_id}:{wake_word}", 0)
                if current_time - last_detection < config.cooldown_seconds:
                    continue
                
                # Simple text matching (can be enhanced with fuzzy matching)
                confidence = self._calculate_text_confidence(wake_word, text_normalized)
                
                if confidence >= config.min_confidence:
                    detection = WakeWordDetection(
                        wake_word=wake_word,
                        confidence=confidence,
                        client_id=client_id,
                        audio_snippet_length=audio_length
                    )
                    
                    # Update cooldown
                    self.last_detection_time[f"{client_id}:{wake_word}"] = current_time
                    
                    # Store recent detection
                    self.recent_detections.append(detection)
                    if len(self.recent_detections) > 50:  # Keep last 50 detections
                        self.recent_detections.pop(0)
                    
                    logger.info(f"Wake word detected: '{wake_word}' from client {client_id} with confidence {confidence:.2f}")
                    
                    # Trigger callbacks
                    self._trigger_callbacks(detection)
                    
                    return detection
        
        return None
    
    def process_audio_raw(self, client_id: str, audio_data, 
                         audio_length: float = 0.0) -> Optional[WakeWordDetection]:
        """
        Process raw audio data for wake word detection.
        
        This is a placeholder for advanced audio-based wake word detection.
        Currently returns None but can be extended with ML-based detection.
        
        Args:
            client_id: ID of the client that provided the audio
            audio_data: Raw audio data (numpy array if available)
            audio_length: Length of the audio snippet in seconds
            
        Returns:
            WakeWordDetection: Detection result if wake word found, None otherwise
        """
        if not self.enabled:
            logger.debug("Raw audio wake word detection skipped - detector disabled")
            return None
        
        # Placeholder for future ML-based wake word detection
        # This could include:
        # - Audio preprocessing (noise reduction, normalization)
        # - Feature extraction (MFCC, spectrograms)
        # - ML model inference for wake word detection
        # - Confidence scoring and thresholding
        
        logger.debug(f"Raw audio wake word detection not yet implemented for client {client_id}")
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
    
    def get_recent_detections(self, limit: int = 10) -> List[WakeWordDetection]:
        """
        Get recent wake word detections.
        
        Args:
            limit: Maximum number of detections to return
            
        Returns:
            List[WakeWordDetection]: Recent detections, most recent first
        """
        with self._lock:
            return list(reversed(self.recent_detections[-limit:]))
    
    def clear_detections(self):
        """Clear all stored detections and reset cooldowns."""
        with self._lock:
            self.recent_detections.clear()
            self.last_detection_time.clear()
            logger.info("Cleared all wake word detections and cooldowns")
    
    def set_enabled(self, enabled: bool):
        """
        Enable or disable the entire wake word detection system.
        
        Args:
            enabled: Whether wake word detection should be enabled
        """
        self.enabled = enabled
        status = "enabled" if enabled else "disabled"
        logger.info(f"Wake word detection {status}")
    
    def get_stats(self) -> Dict[str, Any]:
        """
        Get wake word detection statistics.
        
        Returns:
            Dict[str, Any]: Statistics about wake word detection
        """
        with self._lock:
            total_detections = len(self.recent_detections)
            wake_word_counts = {}
            
            for detection in self.recent_detections:
                wake_word_counts[detection.wake_word] = wake_word_counts.get(detection.wake_word, 0) + 1
            
            return {
                "enabled": self.enabled,
                "total_wake_words": len(self.wake_words),
                "enabled_wake_words": sum(1 for config in self.wake_words.values() if config.enabled),
                "total_detections": total_detections,
                "detections_by_wake_word": wake_word_counts,
                "active_callbacks": len(self.detection_callbacks)
            }