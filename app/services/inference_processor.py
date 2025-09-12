"""
Inference Processor for Action Data Extraction

This module provides action data extraction using Gemini directly with proper dependency injection
and lifecycle management.
"""

import json
from typing import Optional

from google import genai

from app.core.config import settings
from app.core.mira_logger import MiraLogger
from app.models import Action, Interaction


class InferenceProcessor:
    """Inference processor for action data extraction with direct Gemini integration"""

    def __init__(
        self,
        network_id: str,
        system_prompt: str = "You are a helpful AI assistant for action processing.",
        response_format: Optional[dict] = None,
    ):
        """
        Initialize the inference processor with direct Gemini integration.

        Args:
            network_id: ID of the network this processor belongs to
            system_prompt: System prompt for the AI model
            response_format: Response format for structured output
        """
        self.network_id = network_id
        self.system_prompt = system_prompt
        self.response_format = response_format or {}

        self.gemini_client = self._initialize_gemini_client()

        MiraLogger.info(f"InferenceProcessor initialized for network {network_id}")

    def _initialize_gemini_client(self):
        """Initialize Gemini client with API key from settings."""
        api_key = settings.gemini_api_key
        if not api_key:
            raise ValueError("GEMINI_API_KEY environment variable not set")
        return genai.Client(api_key=api_key)

    def _run_gemini_inference(
        self, interaction: Interaction, context: Optional[str] = None
    ) -> dict:
        """Run inference using Gemini API with structured output."""
        from google.genai import types

        system_instruction = self.system_prompt
        if context and context.strip():
            system_instruction += f"\n\nContext: {context.strip()}"

        # Prepare generation config
        config_params = {
            "system_instruction": system_instruction,
            "temperature": 0.7,
            "max_output_tokens": 2048,
        }

        # Add structured output if specified
        if self.response_format:
            config_params["response_mime_type"] = "application/json"
            schema = {
                "type": "object",
                "properties": {
                    key: {"type": "string"} for key in self.response_format.keys()
                },
            }
            config_params["response_json_schema"] = schema

        config = types.GenerateContentConfig(**config_params)

        response = self.gemini_client.models.generate_content(
            model="gemini-1.5-pro", contents=str(interaction.text), config=config
        )

        if not response.text:
            raise RuntimeError("Gemini model generated no content")

        try:
            return json.loads(response.text)
        except json.JSONDecodeError:
            return {"content": response.text}

    def extract_action(
        self, interaction: Interaction, context: Optional[str] = None
    ) -> Action:
        """
        Extract action data using Gemini API with structured output.

        Args:
            interaction: User interaction object
            context: Optional context information

        Returns:
            Action: Extracted action model
        """
        response = self._run_gemini_inference(interaction, context)
        result = Action(**response)
        return result

    def cleanup(self):
        """Clean up resources when the processor is no longer needed."""
        MiraLogger.info(f"Cleaning up InferenceProcessor for network {self.network_id}")
        # Add any cleanup logic here if needed
