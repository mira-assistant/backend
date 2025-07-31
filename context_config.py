from typing import Literal


class ContextProcessorConfig:
    """Configuration class for the context processor."""

    # Speaker Recognition and Clustering
    class SpeakerRecognitionConfig:
        SIMILARITY_THRESHOLD: float = 0.7  # Cosine similarity threshold for same speaker
        DBSCAN_EPS: float = 0.9  # DBSCAN epsilon parameter
        DBSCAN_MIN_SAMPLES: Literal[2] = 2  # DBSCAN minimum samples parameter

    # NLP Settings
    class NLPConfig:
        SPACY_MODEL: Literal["en_core_web_sm"] = "en_core_web_sm"  # spaCy model for NLP
        CONTEXT_SIMILARITY_THRESHOLD: float = 0.7  # Threshold for semantic similarity

    # Context Management
    class ContextManagementParameters:
        CONVERSATION_GAP_THRESHOLD: Literal[300] = 300  # Seconds gap to mark conversation boundary
        SHORT_TERM_CONTEXT_MAX_RESULTS: Literal[20] = 20  # Maximum interactions in short-term context
        LONG_TERM_CONTEXT_MAX_RESULTS: Literal[5] = 5  # Max results for long-term context retrieval
        SUMMARY_MAX_LENGTH: Literal[200] = 200  # Maximum length for context summaries

    # Performance Settings
    class PerformanceConfig:
        BATCH_PROCESSING: Literal[False] = False  # Enable batch processing for better performance
        CACHE_EMBEDDINGS: Literal[True] = True  # Cache voice and text embeddings
        ASYNC_PROCESSING: Literal[False] = False  # Enable asynchronous processing

    # Debug and Logging
    class DebugConfig:
        DEBUG_MODE: Literal[False] = False  # Enable debug logging
        LOG_LEVEL: Literal["INFO"] = "INFO"  # Logging level