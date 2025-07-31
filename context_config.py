from typing import Literal


class ContextProcessorConfig:
    """Configuration class for the context processor."""

    class SpeakerRecognitionConfig:
        """Speaker recognition parameters."""

        SPEAKER_SIMILARITY_THRESHOLD: float = 0.7
        """Cosine similarity threshold for determining if two voice samples are from the same speaker.
        Lowering this value decreases sensitivity, making it less likely to group different speakers together,
        but may increase false negatives (splitting the same speaker into multiple clusters)."""
        DBSCAN_EPS: float = 0.9
        """Epsilon parameter for DBSCAN clustering algorithm, controlling the maximum distance between samples in a cluster."""
        DBSCAN_MIN_SAMPLES: Literal[2] = 2
        """Minimum number of samples required to form a cluster in DBSCAN."""

    class NLPConfig:
        """Natural Language Processing parameters."""

        SPACY_MODEL: Literal["en_core_web_sm"] = "en_core_web_sm"
        """spaCy language model used for natural language processing tasks."""
        CONTEXT_SIMILARITY_THRESHOLD: float = 0.7
        """Threshold for semantic similarity when comparing contexts.
        Lowering this value makes the system more likely to consider contexts as similar,
        potentially increasing recall but reducing precision."""

    class ContextManagementParameters:
        """Parameters for managing context and conversation boundaries."""

        CONVERSATION_GAP_THRESHOLD: Literal[300] = 300
        """Time gap in seconds used to determine conversation boundaries.
        Lowering this value will result in more frequent splitting of conversations."""
        SHORT_TERM_CONTEXT_MAX_RESULTS: Literal[20] = 20
        """Maximum number of recent interactions to include in the short-term context."""
        LONG_TERM_CONTEXT_MAX_RESULTS: Literal[5] = 5
        """Maximum number of results to retrieve from long-term context storage."""
        SUMMARY_MAX_LENGTH: Literal[200] = 200
        """Maximum length (in tokens or characters) for generated context summaries."""

    class PerformanceConfig:
        """Performance optimization parameters."""

        BATCH_PROCESSING: Literal[False] = False
        """Enable or disable batch processing to improve performance on large datasets."""
        CACHE_EMBEDDINGS: Literal[True] = True
        """Enable or disable caching of voice and text embeddings to speed up repeated computations."""
        ASYNC_PROCESSING: Literal[False] = False
        """Enable or disable asynchronous processing for improved throughput."""

    class DebugConfig:
        """Debugging and logging parameters."""

        DEBUG_MODE: Literal[False] = False
        """Enable or disable debug mode for verbose logging and additional diagnostics."""
        LOG_LEVEL: Literal["INFO"] = "INFO"
        """Logging level for controlling the verbosity of log output."""
