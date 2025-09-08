"""
Services package for Mira Backend.

This package contains all the service classes and utilities for the Mira AI Assistant Backend.
"""

# Import all service classes for easy access
from .command_processor import CommandProcessor
from .context_processor import ContextProcessor
from .inference_processor import InferenceProcessor
from .sentence_processor import SentenceProcessor
from .multi_stream_processor import MultiStreamProcessor
from .service_factory import (
    ServiceFactory,
    get_command_processor,
    get_context_processor,
    get_inference_processor,
    get_multi_stream_processor,
    get_sentence_processor,
)
from .service_registry import service_registry

__all__ = [
    "CommandProcessor",
    "ContextProcessor",
    "InferenceProcessor",
    "SentenceProcessor",
    "MultiStreamProcessor",
    "ServiceFactory",
    "get_command_processor",
    "get_context_processor",
    "get_inference_processor",
    "get_multi_stream_processor",
    "get_sentence_processor",
    "service_registry",
]
