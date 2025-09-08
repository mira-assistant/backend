"""
Core module exports.
"""

from .config import settings
from .constants import (
    CONTEXT_SIMILARITY_THRESHOLD,
    CONVERSATION_GAP_THRESHOLD,
    DEFAULT_TIMEOUT,
    MAX_CONTEXT_LENGTH,
    MAX_RETRIES,
    SAMPLE_RATE,
)

__all__ = [
    "settings",
    "SAMPLE_RATE",
    "CONVERSATION_GAP_THRESHOLD",
    "CONTEXT_SIMILARITY_THRESHOLD",
    "MAX_CONTEXT_LENGTH",
    "DEFAULT_TIMEOUT",
    "MAX_RETRIES",
]
