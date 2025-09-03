"""
AI Engine service for managing ML models and inference.
"""

import inspect
import json
import logging
from typing import Dict, List, Optional, Any, Callable
from openai import OpenAI
from openai.types import chat, shared_params

from app.core.config import settings
from app.models.interaction import Interaction

logger = logging.getLogger(__name__)


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
            **config_options: Additional configuration options
        """
        self.model = model_name
        self.system_prompt = system_prompt
        self.tools: list[chat.ChatCompletionToolParam] = []
        self.callables: Dict[str, Callable] = {}
        self.response_format: chat.completion_create_params.ResponseFormat | None = (
            (
                shared_params.ResponseFormatJSONSchema(
                    json_schema=shared_params.response_format_json_schema.JSONSchema(
                        name="Response Model", schema=response_format or {}
                    ),
                    type="json_schema",
                )
            )
            if response_format is not None
            else None
        )

        self.config = {
            **config_options,
        }

        # Initialize OpenAI client
        self.client = OpenAI(base_url=settings.lm_studio_url, api_key=settings.lm_studio_api_key)

        logger.info(f"MLModelManager initialized with model: {model_name}")

    def register_tool(self, function: "Callable", description: str):
        """Register a tool function for the model."""
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

        self.callables[function.__name__] = function
        logger.info(f"Registered tool: {function.__name__}")

    def build_assistant_response(self, response) -> list[chat.ChatCompletionAssistantMessageParam]:
        """Build assistant response with tool calls."""
        tool_calls = response.choices[0].message.tool_calls

        if not tool_calls:
            return []

        assistant_tool_call_message = list[chat.ChatCompletionAssistantMessageParam]()

        assistant_tool_call_message.append(
            chat.ChatCompletionAssistantMessageParam(
                role="assistant",
                tool_calls=(
                    chat.ChatCompletionMessageToolCallParam(
                        id=tool_call.id,
                        type=tool_call.type,
                        function=tool_call.function,
                    )
                    for tool_call in tool_calls
                ),
            )
        )

        for tool_call in tool_calls:
            arguments = (
                json.loads(tool_call.function.arguments)
                if tool_call.function.arguments.strip()
                else {}
            )

            for name, tool in self.callables.items():
                if name == tool_call.function.name:
                    logger.info(f"Invoking tool: {name} with args: {arguments}")
                    function_response = tool(**arguments)
                    logger.info(f"Tool {name} response: {function_response}")

                    assistant_tool_call_message.append(
                        chat.ChatCompletionAssistantMessageParam(
                            role="assistant",
                            name=name,
                            content=function_response,
                        )
                    )

        return assistant_tool_call_message

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

        api_params = {
            "model": self.model,
            "messages": messages,
            "tools": self.tools,
            "tool_choice": "auto",
            **self.config,
        }

        if self.response_format is not None:
            api_params["response_format"] = self.response_format

        response = self.client.chat.completions.create(**api_params, timeout=60)

        if response.choices[0].message.content is None:
            raise RuntimeError(f"Model {self.model} generated no content")

        assistant_message = self.build_assistant_response(response)

        if assistant_message:
            messages.extend(assistant_message)
            response = self.client.chat.completions.create(**api_params)

        content = response.choices[0].message.content
        if content:
            try:
                return json.loads(content)
            except json.JSONDecodeError:
                raise  # Re-raise the JSONDecodeError
        return response


def get_available_models() -> List[Dict[str, Any]]:
    """
    Get list of available models from LM Studio server

    Returns:
        List[Dict]: List of available model information
    """
    try:
        client = OpenAI(base_url=settings.lm_studio_url, api_key=settings.lm_studio_api_key)
        response = client.models.list()
        data = response.model_dump()["data"]
        return data
    except Exception as e:
        logger.error(f"Failed to get available models: {e}")
        raise RuntimeError(f"Could not fetch available models: {e}")
