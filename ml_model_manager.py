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

import requests
import json
import logging
from typing import Dict, List, Optional, Any, Callable
from dataclasses import dataclass
from datetime import datetime

logger = logging.getLogger(__name__)


@dataclass
class ModelResponse:
    """Response from ML model with structured data"""
    
    callback_function: Optional[str] = None
    callback_arguments: Dict[str, Any] = None
    action_data: Optional[Dict[str, Any]] = None
    action_type: Optional[str] = None
    user_response: str = ""
    success: bool = True
    error: Optional[str] = None
    
    def __post_init__(self):
        if self.callback_arguments is None:
            self.callback_arguments = {}


class MLModelManager:
    """
    Manages all interactions with LM Studio server for AI model communication.
    
    This class provides a unified interface for both command inference and action data
    extraction, with support for structured prompts and callback function management.
    """
    
    # Static configuration
    LM_STUDIO_URL = "http://localhost:1234/v1"
    CHAT_ENDPOINT = f"{LM_STUDIO_URL}/chat/completions"
    MODELS_ENDPOINT = f"{LM_STUDIO_URL}/models"
    
    def __init__(self, model_name: str = "microsoft/DialoGPT-small"):
        """
        Initialize ML Model Manager
        
        Args:
            model_name: Name of the model to use for inference
        """
        self.model = model_name
        self.payload_config = {
            "max_tokens": -1,
            "stream": False,
            "temperature": 0.3,
            "top_p": 0.8,
            "top_k": 40,
            "repeat_penalty": 1.2,
            "min_p": 0.2,
        }
        
        # System prompts as state variables
        self.command_system_prompt = self._build_command_system_prompt()
        self.action_system_prompt = self._build_action_system_prompt()
        
        logger.info(f"MLModelManager initialized with model: {model_name}")
    
    def _build_command_system_prompt(self) -> str:
        """Build system prompt for command inference"""
        return """You are Mira, an AI assistant that processes voice commands and determines which callback function (if any) should be invoked.

Your task:
1. Analyze the user input to determine if a callback function should be invoked
2. If a callback is needed, identify which function and any required arguments
3. Provide a user-facing response sentence
4. Support recursive callback execution by including callback context when needed

Respond with JSON in this format:
{
    "callback_function": "function_name" or null,
    "callback_arguments": {"arg1": "value1", "arg2": "value2"} or {},
    "user_response": "What to say back to the user"
}

Available Functions will be provided in the user prompt.

Examples:
- "What time is it?" -> {"callback_function": "getTime", "callback_arguments": {}, "user_response": ""}
- "What's the weather like?" -> {"callback_function": "getWeather", "callback_arguments": {"location": "current location"}, "user_response": ""}
- "Stop listening" -> {"callback_function": "disableMira", "callback_arguments": {}, "user_response": ""}
- "Hello there" -> {"callback_function": null, "callback_arguments": {}, "user_response": "Hello! How can I help you?"}

Only invoke callbacks for clear, actionable requests. Respond naturally and conversationally."""
    
    def _build_action_system_prompt(self) -> str:
        """Build system prompt for action data extraction"""
        return """You are Mira, an AI assistant that extracts structured action data from user interactions.

Your task:
1. Analyze the user input to identify actionable items (calendar entries, reminders, etc.)
2. Extract relevant data such as times, names, descriptions, locations
3. Determine the action type (calendar_entry, reminder, note, etc.)
4. Structure the data in a consistent format

Respond with JSON in this format:
{
    "action_type": "calendar_entry" | "reminder" | "note" | "task" | null,
    "action_data": {
        "title": "Event/task title",
        "description": "Detailed description", 
        "start_time": "ISO timestamp or null",
        "end_time": "ISO timestamp or null",
        "location": "Location string or null",
        "priority": "high" | "medium" | "low" | null,
        "participants": ["person1", "person2"] or []
    },
    "user_response": "Confirmation message for the user"
}

Examples:
- "Schedule a meeting with John tomorrow at 2 PM" -> Extract calendar_entry with appropriate data
- "Remind me to call mom at 5" -> Extract reminder with time and description
- "Add a note about the quarterly report" -> Extract note with description

Focus on extracting accurate time information and relevant details."""
    
    @staticmethod
    def get_available_models() -> List[Dict[str, Any]]:
        """
        Get list of available models from LM Studio server
        
        Returns:
            List[Dict]: List of available model information
        """
        try:
            response = requests.get(MLModelManager.MODELS_ENDPOINT)
            response.raise_for_status()
            data = response.json()
            return data.get("data", [])
        except Exception as e:
            logger.error(f"Failed to get available models: {e}")
            raise
    
    def _send_request(self, messages: List[Dict[str, str]]) -> Dict[str, Any]:
        """
        Send request to LM Studio server
        
        Args:
            messages: List of message objects for the conversation
            
        Returns:
            Dict: Response from the model
        """
        payload = {
            "model": self.model,
            "messages": messages,
            **self.payload_config
        }
        
        response = requests.post(self.CHAT_ENDPOINT, json=payload)
        response.raise_for_status()
        data = response.json()
        
        generated_text = data.get("choices", [{}])[0].get("message", {}).get("content", "")
        
        # Parse JSON response
        result = json.loads(generated_text)
        return result
    
    def process_command_inference(
        self,
        interaction_text: str,
        available_functions: List[str],
        function_descriptions: Dict[str, str],
        context: Optional[str] = None
    ) -> ModelResponse:
        """
        Process command inference to determine callback functions
        
        Args:
            interaction_text: User interaction text
            available_functions: List of available callback function names
            function_descriptions: Mapping of function names to descriptions
            context: Optional context from previous callback executions
            
        Returns:
            ModelResponse: Structured response with callback information
        """
        # Build function list for prompt
        function_list = ""
        for func_name in available_functions:
            description = function_descriptions.get(func_name, "No description available")
            function_list += f"\n- {func_name}: {description}"
        
        # Build user prompt
        user_prompt = f"""Available Functions:{function_list}

User Input: {interaction_text}"""
        
        if context:
            user_prompt += f"\n\nPrevious Context:\n{context}"
        
        messages = [
            {"role": "system", "content": self.command_system_prompt},
            {"role": "user", "content": user_prompt}
        ]
        
        # Let exceptions propagate - server should crash when ML model is offline
        result = self._send_request(messages)
        
        return ModelResponse(
            callback_function=result.get("callback_function"),
            callback_arguments=result.get("callback_arguments", {}),
            user_response=result.get("user_response", ""),
            success=True
        )
    
    def process_action_extraction(
        self,
        interaction_text: str,
        context: Optional[str] = None
    ) -> ModelResponse:
        """
        Process action data extraction from user interaction
        
        Args:
            interaction_text: User interaction text
            context: Optional context information
            
        Returns:
            ModelResponse: Structured response with action data
        """
        user_prompt = f"User Input: {interaction_text}"
        
        if context:
            user_prompt += f"\n\nContext:\n{context}"
        
        messages = [
            {"role": "system", "content": self.action_system_prompt},
            {"role": "user", "content": user_prompt}
        ]
        
        # Let exceptions propagate - server should crash when ML model is offline
        result = self._send_request(messages)
        
        return ModelResponse(
            action_type=result.get("action_type"),
            action_data=result.get("action_data", {}),
            user_response=result.get("user_response", ""),
            success=True
        )
    
    def process_recursive_command(
        self,
        interaction_text: str,
        available_functions: List[str],
        function_descriptions: Dict[str, str],
        callback_executor: Callable[[str, Dict[str, Any]], Any],
        max_recursion: int = 3
    ) -> ModelResponse:
        """
        Process command with recursive callback execution
        
        Args:
            interaction_text: User interaction text
            available_functions: List of available callback function names
            function_descriptions: Mapping of function names to descriptions
            callback_executor: Function to execute callbacks
            max_recursion: Maximum recursion depth
            
        Returns:
            ModelResponse: Final response after all recursive calls
        """
        context = ""
        recursion_count = 0
        final_response = None
        
        current_text = interaction_text
        
        while recursion_count < max_recursion:
            # Get model response
            response = self.process_command_inference(
                current_text,
                available_functions,
                function_descriptions,
                context if context else None
            )
            
            if not response.success:
                return response
            
            final_response = response
            
            # If no callback function, we're done
            if not response.callback_function:
                break
            
            # Execute callback
            try:
                callback_result = callback_executor(
                    response.callback_function,
                    response.callback_arguments
                )
                
                # Add to context for potential next iteration
                context += f"\nExecuted {response.callback_function} with result: {callback_result}"
                
                # Check if the result suggests another action
                if isinstance(callback_result, str) and len(callback_result.strip()) > 0:
                    # Use callback result as input for potential next iteration
                    current_text = callback_result
                    recursion_count += 1
                else:
                    break
                    
            except Exception as e:
                logger.error(f"Callback execution failed in recursive processing: {e}")
                return ModelResponse(
                    success=False,
                    error=f"Callback execution failed: {e}",
                    user_response="I encountered an error while processing that command."
                )
        
        return final_response
    
    def set_model(self, model_name: str):
        """
        Change the active model
        
        Args:
            model_name: Name of the new model to use
        """
        self.model = model_name
        logger.info(f"Changed model to: {model_name}")
    
    def update_payload_config(self, **kwargs):
        """
        Update payload configuration parameters
        
        Args:
            **kwargs: Key-value pairs to update in payload config
        """
        self.payload_config.update(kwargs)
        logger.info(f"Updated payload config: {kwargs}")