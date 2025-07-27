"""
Configuration system for the enhanced context processor.
"""

from dataclasses import dataclass
from typing import Optional, Dict, Any
import json
import os


@dataclass
class ContextProcessorConfig:
    """Configuration class for the enhanced context processor."""
    
    # Conversation Management
    short_term_window: int = 10  # Time window in seconds for short-term context (fallback)
    max_history_size: int = 1000  # Maximum number of interactions to keep in history
    max_conversation_length: int = 20  # Maximum interactions in short-term context
    conversation_gap_threshold: int = 300  # Seconds gap to mark conversation boundary
    
    # Speaker Recognition and Clustering
    similarity_threshold: float = 0.75  # Cosine similarity threshold for same speaker
    max_speakers: int = 10  # Maximum number of speakers to track
    dbscan_eps: float = 0.3  # DBSCAN epsilon parameter
    dbscan_min_samples: int = 2  # DBSCAN minimum samples parameter
    voice_embedding_update_rate: float = 0.1  # Rate for updating voice embeddings
    
    # NLP Settings
    enable_ner: bool = True  # Enable Named Entity Recognition
    enable_coreference: bool = True  # Enable coreference resolution
    enable_topic_modeling: bool = True  # Enable topic modeling
    enable_sentiment_analysis: bool = True  # Enable sentiment analysis
    spacy_model: str = "en_core_web_sm"  # spaCy model for NLP
    max_topics: int = 5  # Maximum topics for topic modeling
    
    # Context Retrieval
    long_term_context_max_results: int = 5  # Max results for long-term context retrieval
    context_similarity_threshold: float = 0.7  # Threshold for semantic similarity
    enable_context_summarization: bool = True  # Enable context summarization
    summary_max_length: int = 200  # Maximum length for context summaries
    
    # Performance Settings
    batch_processing: bool = False  # Enable batch processing for better performance
    cache_embeddings: bool = True  # Cache voice and text embeddings
    async_processing: bool = False  # Enable asynchronous processing
    
    # Debug and Logging
    debug_mode: bool = False  # Enable debug logging
    log_level: str = "INFO"  # Logging level
    save_intermediate_results: bool = False  # Save intermediate processing results
    
    @classmethod
    def load_from_file(cls, config_path: str) -> "ContextProcessorConfig":
        """Load configuration from a JSON file."""
        if os.path.exists(config_path):
            with open(config_path, 'r') as f:
                config_data = json.load(f)
            return cls(**config_data)
        return cls()
    
    def save_to_file(self, config_path: str) -> None:
        """Save configuration to a JSON file."""
        with open(config_path, 'w') as f:
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


# Default configuration instance
DEFAULT_CONFIG = ContextProcessorConfig()


def get_default_config() -> ContextProcessorConfig:
    """Get the default configuration instance."""
    return DEFAULT_CONFIG


def create_config(**kwargs) -> ContextProcessorConfig:
    """Create a new configuration with custom parameters."""
    config = ContextProcessorConfig()
    config.update(**kwargs)
    return config