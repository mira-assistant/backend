"""
Inference service for action extraction.
"""

import json
import logging
from app.services.ai_engine import MLModelManager
from app.models.action import Action
from app.models.interaction import Interaction

logger = logging.getLogger(__name__)


class InferenceProcessor:
    """Inference processor for action extraction."""

    def __init__(self):
        """Initialize the inference processor."""
        try:
            with open("schemas/action_processing/system_prompt.txt", "r") as f:
                system_prompt = f.read().strip()
        except FileNotFoundError:
            system_prompt = "You are an AI assistant that extracts actions from user interactions."

        try:
            with open("schemas/action_processing/structured_output.json", "r") as f:
                structured_response = json.load(f)
        except FileNotFoundError:
            structured_response = {}

        self.model_manager = MLModelManager(
            "tiiuae-falcon-40b-instruct", system_prompt, structured_response
        )

        logging.info("InferenceProcessor initialized")

    def extract_action(self, interaction: Interaction, context=None) -> Action:
        """
        Extract action from user interaction.

        Args:
            interaction: User interaction object
            context: Optional context information

        Returns:
            Action: Extracted action model
        """
        response = self.model_manager.run_inference(interaction, context)
        result = Action(**response)
        return result
