"""
Inference Processor for Action Data Extraction

This module provides action data extraction using AI models with proper dependency injection
and lifecycle management.
"""

import logging
from app.services.ml_model_manager import MLModelManager
from app.models import Action, Interaction


class InferenceProcessor:
    """Inference processor for action data extraction with proper dependency injection"""

    def __init__(self, model_manager: MLModelManager, network_id: str):
        """
        Initialize the inference processor with dependency injection.

        Args:
            model_manager: Injected ML model manager
            network_id: ID of the network this processor belongs to
        """
        self.model_manager = model_manager
        self.network_id = network_id

        logging.info(f"InferenceProcessor initialized for network {network_id}")

    def extract_action(self, interaction: Interaction, context=None) -> Action:
        """
        Sends a prompt to the LM Studio API for action data extraction.
        This function is maintained for backward compatibility but now uses
        the ML model manager for structured action extraction.

        Args:
            interaction: User interaction object
            context: Optional context information

        Returns:
            Action: Extracted action model
        """
        response = self.model_manager.run_inference(interaction, context)
        result = Action(**response)
        return result

    def cleanup(self):
        """Clean up resources when the processor is no longer needed."""
        logging.info(f"Cleaning up InferenceProcessor for network {self.network_id}")
        # Add any cleanup logic here if needed
