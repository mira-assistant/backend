"""
Service Factory for Network-Specific Service Creation

This module provides factory functions for creating network-specific service instances
with proper dependency injection and configuration.
"""

from typing import Any, Dict

from core.constants import SAMPLE_RATE
from core.mira_logger import MiraLogger

# MLModelManager no longer needed - using direct Gemini integration
from services.service_registry import service_registry


class ServiceFactory:
    """Factory for creating network-specific services with proper dependency injection."""

    @staticmethod
    def create_command_processor(network_id: str):
        """Create a CommandProcessor for a specific network."""
        from services.command_processor import CommandProcessor

        # Load network-specific configuration
        config = ServiceFactory._load_network_config(network_id)

        # Create command processor with direct Gemini integration
        return CommandProcessor(
            network_id=network_id,
            system_prompt=config.get(
                "system_prompt", ServiceFactory._load_default_system_prompt()
            ),
        )

    @staticmethod
    def create_context_processor(network_id: str):
        """Create a ContextProcessor for a specific network."""
        from services.context_processor import ContextProcessor

        # Load network-specific configuration
        config = ServiceFactory._load_network_config(network_id)

        return ContextProcessor(network_id, config)

    @staticmethod
    def create_inference_processor(network_id: str):
        """Create an InferenceProcessor for a specific network."""
        from services.inference_processor import InferenceProcessor

        # Load network-specific configuration
        config = ServiceFactory._load_network_config(network_id)

        # Create inference processor with direct Gemini integration
        return InferenceProcessor(
            network_id=network_id,
            system_prompt=config.get(
                "system_prompt", ServiceFactory._load_action_system_prompt()
            ),
            response_format=config.get(
                "response_format", ServiceFactory._load_action_response_format()
            ),
        )

    @staticmethod
    def create_multi_stream_processor(network_id: str):
        """Create a MultiStreamProcessor for a specific network."""
        from services.multi_stream_processor import MultiStreamProcessor

        # Load network-specific configuration
        config = ServiceFactory._load_network_config(network_id)

        return MultiStreamProcessor(network_id, config)

    @staticmethod
    def create_sentence_processor(network_id: str):
        """Create a SentenceProcessor for a specific network."""
        from services.sentence_processor import SentenceProcessor

        # Load network-specific configuration
        config = ServiceFactory._load_network_config(network_id)

        return SentenceProcessor(network_id, config)

    @staticmethod
    def _load_network_config(network_id: str) -> Dict[str, Any]:
        """Load network-specific configuration."""
        # For now, return default config
        # In production, this would load from database or config service
        return {
            "model_name": "gemini-1.5-pro",
            "client_system": "gemini",
            "system_prompt": ServiceFactory._load_default_system_prompt(),
            "wake_words": [
                {"word": "mira", "sensitivity": 0.7, "min_confidence": 0.5},
                {"word": "hey mira", "sensitivity": 0.8, "min_confidence": 0.6},
            ],
            "sample_rate": SAMPLE_RATE,
            "max_clients": 10,
        }

    @staticmethod
    def _load_default_system_prompt() -> str:
        """Load the default system prompt for command processing."""
        try:
            with open("schemas/command_processing/system_prompt.txt", "r") as f:
                return f.read().strip()
        except FileNotFoundError:
            MiraLogger.warning(
                "Command processing system prompt not found, using default"
            )
            return "You are a helpful AI assistant."

    @staticmethod
    def _load_action_system_prompt() -> str:
        """Load the system prompt for action processing."""
        try:
            with open("schemas/action_processing/system_prompt.txt", "r") as f:
                return f.read().strip()
        except FileNotFoundError:
            MiraLogger.warning(
                "Action processing system prompt not found, using default"
            )
            return "You are a helpful AI assistant for action processing."

    @staticmethod
    def _load_action_response_format() -> Dict[str, Any]:
        """Load the response format for action processing."""
        try:
            import json

            with open("schemas/action_processing/structured_output.json", "r") as f:
                return json.load(f)
        except FileNotFoundError:
            MiraLogger.warning(
                "Action processing response format not found, using default"
            )
            return {}


# Tool registration is now handled directly in CommandProcessor and InferenceProcessor


# Convenience functions for getting services
def get_command_processor(network_id: str):
    """Get a CommandProcessor for the specified network."""
    return service_registry.get_service(
        network_id, "command_processor", ServiceFactory.create_command_processor
    )


def get_context_processor(network_id: str):
    """Get a ContextProcessor for the specified network."""
    return service_registry.get_service(
        network_id, "context_processor", ServiceFactory.create_context_processor
    )


def get_inference_processor(network_id: str):
    """Get an InferenceProcessor for the specified network."""
    return service_registry.get_service(
        network_id, "inference_processor", ServiceFactory.create_inference_processor
    )


def get_multi_stream_processor(network_id: str):
    """Get a MultiStreamProcessor for the specified network."""
    return service_registry.get_service(
        network_id,
        "multi_stream_processor",
        ServiceFactory.create_multi_stream_processor,
    )


def get_sentence_processor(network_id: str):
    """Get a SentenceProcessor for the specified network."""
    return service_registry.get_service(
        network_id, "sentence_processor", ServiceFactory.create_sentence_processor
    )
