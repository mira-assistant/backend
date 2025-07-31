"""
Configuration system for the context processor.
"""

from dataclasses import dataclass
from typing import Dict, Any
import json
import os


@dataclass
class ContextProcessorConfig:
    """Configuration class for the context processor."""

    # Speaker Recognition and Clustering
    class SpeakerRecognitionConfig:
        SIMILARITY_THRESHOLD: float = 0.7  # Cosine similarity threshold for same speaker
        DBSCAN_EPS: float = 0.9  # DBSCAN epsilon parameter
        DBSCAN_MIN_SAMPLES: int = 2  # DBSCAN minimum samples parameter

    # NLP Settings
    class NLPConfig:
        SPACY_MODEL: str = "en_core_web_sm"  # spaCy model for NLP
        CONTEXT_SIMILARITY_THRESHOLD: float = 0.7  # Threshold for semantic similarity

    # Context Management
    class ContextManagementParameters:
        CONVERSATION_GAP_THRESHOLD: int = 300  # Seconds gap to mark conversation boundary
        SHORT_TERM_CONTEXT_MAX_RESULTS: int = 20  # Maximum interactions in short-term context
        LONG_TERM_CONTEXT_MAX_RESULTS: int = 5  # Max results for long-term context retrieval
        SUMMARY_MAX_LENGTH: int = 200  # Maximum length for context summaries

    # Performance Settings
    class PerformanceConfig:
        BATCH_PROCESSING: bool = False  # Enable batch processing for better performance
        CACHE_EMBEDDINGS: bool = True  # Cache voice and text embeddings
        ASYNC_PROCESSING: bool = False  # Enable asynchronous processing

    # Debug and Logging
    class DebugConfig:
        DEBUG_MODE: bool = False  # Enable debug logging
        LOG_LEVEL: str = "INFO"  # Logging level

    @classmethod
    def load_from_file(cls, config_path: str) -> "ContextProcessorConfig":
        """Load configuration from a JSON file."""
        if os.path.exists(config_path):
            with open(config_path, "r") as f:
                config_data = json.load(f)
            return cls(**config_data)
        return cls()

    def save_to_file(self, config_path: str) -> None:
        """Save configuration to a JSON file."""
        with open(config_path, "w") as f:
            json.dump(self.__dict__, f, indent=2)

    def update(self, **kwargs) -> None:
        """Update configuration parameters."""
        for key, value in kwargs.items():
            if hasattr(self, key):
                setattr(self, key, value)
            else:
                raise ValueError(f"Unknown configuration parameter: {key}")

    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary."""
        return self.__dict__.copy()
