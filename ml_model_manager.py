"""
ML Model Manager

This module manages all interactions with the LM Studio server for both command inference
and action data extraction. It provides a unified interface for AI model communication
with support for structured prompts, callback functions, and recursive execution.

Features:
- Command inference with callback function support
- Action data extraction for calendar entries, etc.
- Structured prompt management as state variables
- Recursive callback execution with context
- Model management and availability checking
- Configurable model and payload parameters
"""

import inspect
import json
import logging
from typing import Dict, List, Optional, Any, Callable
from models import Interaction


logger = logging.getLogger(__name__)

LM_STUDIO_URL = "http://localhost:1234/v1"
client = None


def get_openai_client():
    """Get OpenAI client with timeout configuration"""
    global client
    if client is None:
        from openai import OpenAI
        client = OpenAI(
            base_url=LM_STUDIO_URL, 
            api_key="lm-studio",
            timeout=5.0,  # 5 second timeout
            max_retries=0  # No retries to avoid hanging
        )
    return client


class MLModelManager:
    """
    Individual ML Model configuration with system prompt, endpoint, and inference capability
    """

    def __init__(
        self,
        model_name: str,
        system_prompt: str,
        response_format: Optional[dict] = None,
        **config_options,
    ):
        """
        Initialize ML Model

        Args:
            model_name: Name of the model to use for inference
            system_prompt: Custom system prompt or uses default
            response_format: Optional JSON schema for structured output
            **config_options: Additional configuration options including:
                - temperature: Sampling temperature (0.0-2.0), default 0.7
                - max_tokens: Maximum number of tokens to generate
                - top_k: Top-k sampling parameter
                - top_p: Top-p sampling parameter
                - repetition_penalty: Repetition penalty
                - frequency_penalty: Frequency penalty
                - presence_penalty: Presence penalty
        """

        # Don't check models during initialization to avoid blocking
        self.model = model_name
        self.system_prompt = system_prompt
        self.tools: list = []

        # Initialize response format
        self.response_format = None
        if response_format is not None:
            from openai.types import chat, shared_params
            self.response_format = shared_params.ResponseFormatJSONSchema(
                json_schema=shared_params.response_format_json_schema.JSONSchema(
                    name="Response Model", schema=response_format
                ),
                type="json_schema",
            )

        self.config = {
            **config_options,
        }

        logger.info(f"MLModelManager initialized with model: {model_name}")

    def register_tool(self, function: 'Callable', description: str):
        # Create a basic function definition without complex parameter introspection
        tool_definition = {
            "type": "function",
            "function": {
                "name": function.__name__,
                "description": description,
                "parameters": {"type": "object", "properties": {}, "required": []},
            }
        }
        self.tools.append(tool_definition)
        logger.info(f"Registered tool: {function.__name__}")

    def run_inference(
        self, interaction: Interaction, context: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Run inference on the model

        Args:
            interaction: User interaction text
            context: Optional context information

        Returns:
            Dict: Response from the model
        """
        
        client = get_openai_client()
        from openai.types import chat
        
        messages: list = []

        messages.append({
            "content": self.system_prompt,
            "role": "system",
        })

        if context and context.strip():
            messages.append({
                "content": f"Context: {context.strip()}", 
                "role": "assistant", 
                "name": "context_provider"
            })

        messages.append({
            "content": interaction.text,
            "role": "user",
        })

        api_params = {
            "model": self.model,
            "messages": messages,
            "tools": self.tools,
            "tool_choice": "auto",
            **self.config,
        }

        if self.response_format is not None:
            api_params["response_format"] = self.response_format

        response = client.chat.completions.create(**api_params)

        if response.choices[0].message.content is None:
            raise RuntimeError(f"Model {self.model} generated no content")

        results = json.loads(response.choices[0].message.content)
        return results


def get_available_models() -> List[Dict[str, Any]]:
    """
    Get list of available models from LM Studio server

    Returns:
        List[Dict]: List of available model information
    """
    client = get_openai_client()
    response = client.models.list()
    data = response.model_dump()["data"]
    return data
