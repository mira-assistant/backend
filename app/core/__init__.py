"""
Core module exports.
"""

from .config import settings
from .constants import (
    SAMPLE_RATE,
    CONVERSATION_GAP_THRESHOLD,
    CONTEXT_SIMILARITY_THRESHOLD,
    MAX_CONTEXT_LENGTH,
    DEFAULT_TIMEOUT,
    MAX_RETRIES,
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
