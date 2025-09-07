# """
# ML Model Manager - Simple and Clean Implementation
# """

# import json
# import logging
# import os
# from typing import Dict, List, Optional, Any, Callable
# from enum import Enum

# from app.models import Interaction
# from openai import OpenAI
# from google import genai
# from openai.types import chat

# logger = logging.getLogger(__name__)


# class ClientSystem(Enum):
#     """Supported client systems."""

#     GEMINI = "gemini"
#     OPENAI = "openai"
#     LM_STUDIO = "lm_studio"


# class MLModelManager:
#     """Simple ML Model Manager with backend switching."""

#     def __init__(
#         self,
#         model_name: str,
#         client_system: ClientSystem,
#         system_prompt: str = "You are a helpful AI assistant.",
#         response_format: Optional[Dict[str, str]] = None,
#         temperature: float = 0.7,
#         max_tokens: int = 2048,
#     ):
#         self.model = model_name
#         self.client_system = client_system
#         self.system_prompt = system_prompt
#         self.response_format = response_format
#         self.temperature = temperature
#         self.max_tokens = max_tokens
#         self.tools = []

#         # Initialize client based on system
#         self.client = self._initialize_client()

#         logger.info(
#             f"MLModelManager initialized with {client_system.value} client, model: {self.model}"
#         )

#     def _initialize_client(self):
#         """Initialize the appropriate client based on client_system."""
#         if self.client_system == ClientSystem.GEMINI:
#             api_key = os.getenv("GEMINI_API_KEY")
#             if not api_key:
#                 raise ValueError("GEMINI_API_KEY environment variable not set")
#             return genai.Client(api_key=api_key)

#         elif self.client_system == ClientSystem.OPENAI:
#             api_key = os.getenv("OPENAI_API_KEY")
#             if not api_key:
#                 raise ValueError("OPENAI_API_KEY environment variable not set")
#             return OpenAI(api_key=api_key)

#         elif self.client_system == ClientSystem.LM_STUDIO:
#             api_key = os.getenv("LM_STUDIO_API_KEY", "lm-studio")
#             return OpenAI(base_url="http://localhost:1234/v1", api_key=api_key)

#         else:
#             raise ValueError(f"Unsupported client system: {self.client_system}")

#     def register_tool(self, function: Callable, description: str):
#         """Register a tool function."""
#         self.tools.append(
#             {"function": function, "description": description, "name": function.__name__}
#         )

#     def run_inference(
#         self, interaction: Interaction, context: Optional[str] = None
#     ) -> Dict[str, Any]:
#         """Run inference with the configured client system."""
#         try:
#             if self.client_system == ClientSystem.GEMINI:
#                 return self._run_gemini_inference(interaction, context)
#             else:
#                 return self._run_openai_inference(interaction, context)
#         except Exception as e:
#             logger.error(f"Inference failed for {self.client_system.value}: {e}")
#             raise

#     def _run_gemini_inference(
#         self, interaction: Interaction, context: Optional[str]
#     ) -> Dict[str, Any]:
#         """Run inference using Gemini API."""
#         from google.genai import types

#         gemini_client = self.client

#         system_instruction = self.system_prompt
#         if context and context.strip():
#             system_instruction += f"\n\nContext: {context.strip()}"

#         # Prepare generation config
#         config_params = {
#             "system_instruction": system_instruction,
#             "temperature": self.temperature,
#             "max_output_tokens": self.max_tokens,
#         }

#         # Add structured output if specified
#         if self.response_format:
#             config_params["response_mime_type"] = "application/json"
#             # Create a simple JSON schema from response_format
#             if isinstance(self.response_format, dict):
#                 schema = {
#                     "type": "object",
#                     "properties": {
#                         key: {"type": "string"} for key in self.response_format.keys()
#                     }
#                 }
#                 config_params["response_json_schema"] = schema

#         # Add tools if any are registered
#         if self.tools:
#             # Convert tools to Gemini format
#             gemini_tools = []
#             for tool in self.tools:
#                 # For now, we'll create a simple function declaration
#                 # In a full implementation, you'd convert the tool function to Gemini's format
#                 gemini_tools.append({
#                     "function_declarations": [{
#                         "name": tool["name"],
#                         "description": tool["description"],
#                         "parameters": {
#                             "type": "object",
#                             "properties": {
#                                 "query": {"type": "string", "description": "The query parameter"}
#                             },
#                             "required": ["query"]
#                         }
#                     }]
#                 })
#             config_params["tools"] = gemini_tools

#         config = types.GenerateContentConfig(**config_params)

#         response = gemini_client.models.generate_content(
#             model=self.model,
#             contents=str(interaction.text),
#             config=config
#         )

#         if not response.text:
#             raise RuntimeError(f"Gemini model {self.model} generated no content")

#         try:
#             return json.loads(response.text)
#         except json.JSONDecodeError:
#             return {"content": response.text}

#     def _run_openai_inference(
#         self, interaction: Interaction, context: Optional[str]
#     ) -> Dict[str, Any]:
#         """Run inference using OpenAI/LM Studio API."""
#         client = self.client

#         system_content = self.system_prompt
#         if context and context.strip():
#             system_content += f"\n\nContext: {context.strip()}"

#         messages = [
#             chat.ChatCompletionSystemMessageParam(role="system", content=system_content),
#             chat.ChatCompletionUserMessageParam(role="user", content=str(interaction.text)),
#         ]

#         # Add structured output if specified
#         kwargs = {
#             "model": self.model,
#             "messages": messages,
#             "temperature": self.temperature,
#             "max_tokens": self.max_tokens,
#         }

#         if self.response_format:
#             kwargs["response_format"] = {"type": "json_object"}

#         response = client.chat.completions.create(**kwargs)

#         content = response.choices[0].message.content
#         if not content:
#             raise RuntimeError(f"Model {self.model} generated no content")

#         try:
#             return json.loads(content)
#         except json.JSONDecodeError:
#             return {"content": content}

#     def get_available_models(self) -> List[Dict[str, Any]]:
#         """Get available models for the current client system."""
#         try:
#             if self.client_system == ClientSystem.GEMINI:
#                 return self._get_gemini_models()
#             else:
#                 return self._get_openai_models()
#         except Exception as e:
#             logger.error(f"Failed to get available models: {e}")
#             raise RuntimeError(f"Could not fetch available models: {e}")

#     def _get_gemini_models(self) -> List[Dict[str, Any]]:
#         """Get available Gemini models."""
#         genai = self.client
#         models = genai.list_models()  # type: ignore

#         model_list = []
#         for model in models:
#             if "generateContent" in model.supported_generation_methods:
#                 model_name = model.name or "unknown"
#                 model_list.append(
#                     {
#                         "id": model_name.split("/")[-1],
#                         "state": "loaded",
#                         "name": model_name,
#                     }
#                 )
#         return model_list

#     def _get_openai_models(self) -> List[Dict[str, Any]]:
#         """Get available OpenAI/LM Studio models."""
#         client = self.client
#         response = client.models.list()
#         return response.model_dump()["data"]


# # Backward compatibility - deprecated, use MLModelManager directly
# def get_available_models() -> List[Dict[str, Any]]:
#     """Get available models for the configured backend."""
#     # Default to Gemini for backward compatibility
#     temp_manager = MLModelManager(model_name="gemini-1.5-pro", client_system=ClientSystem.GEMINI)
#     return temp_manager.get_available_models()
