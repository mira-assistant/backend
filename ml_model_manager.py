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
from openai import OpenAI
from openai.types import chat, shared_params
from models import Interaction


logger = logging.getLogger(__name__)

LM_STUDIO_URL = "http://localhost:1234/v1"
client = OpenAI(base_url=LM_STUDIO_URL, api_key="lm-studio")


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

        available_models = get_available_models()

        model_names = [model.get("id", "") for model in available_models]
        model_states = [model.get("state", "") for model in available_models]

        # if model_name not in model_names or model_states[model_names.index(model_name)] != "loaded":
        #     raise ValueError(f"Model '{model_name}' not available or loaded")

        self.model = model_name
        self.system_prompt = system_prompt
        self.tools: list[chat.ChatCompletionToolParam] = []

        if response_format is not None:
            self.response_format: chat.completion_create_params.ResponseFormat = (
                shared_params.ResponseFormatJSONSchema(
                    json_schema=shared_params.response_format_json_schema.JSONSchema(
                        name="Response Model", schema=response_format
                    ),
                    type="json_schema",
                )
            )
        else:
            self.response_format: chat.completion_create_params.ResponseFormat = (
                shared_params.ResponseFormatText(type="text")
            )

        self.config = {
            **config_options,
        }

        logger.info(f"MLModelManager initialized with model: {model_name}")

    def register_tool(self, function: 'Callable', description: str):
        parameters: shared_params.FunctionParameters = {**inspect.signature(function).parameters}

        self.tools.append(
            chat.ChatCompletionToolParam(
                function=shared_params.FunctionDefinition(
                    name=function.__name__,
                    description=description,
                    parameters=parameters,
                ),
                type="function",
            )
        )

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

        messages: list[chat.ChatCompletionMessageParam] = list()

        messages.append(
            chat.ChatCompletionSystemMessageParam(
                content=self.system_prompt,
                role="system",
            )
        )

        # Add context as a separate assistant role if provided for better separation
        if context and context.strip():
            messages.append(
                chat.ChatCompletionAssistantMessageParam(
                    content=f"Context: {context.strip()}", role="assistant", name="context_provider"
                )
            )

        messages.append(
            chat.ChatCompletionUserMessageParam(
                content=interaction.text,  # type: ignore
                role="user",
            )
        )

        # Prepare the API call parameters
        api_params = {
            "model": self.model,
            "messages": messages,
            "tools": self.tools,
            "tool_choice": "auto",
            **self.config,
        }

        # Only include response_format if it's not None
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
    try:
        response = client.models.list()
        data = response.model_dump()["data"]

        return data
    except Exception as e:
        logger.error(f"Failed to get available models: {e}")
        raise RuntimeError(f"Could not fetch available models: {e}")
