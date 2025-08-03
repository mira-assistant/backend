"""
Command Processing Workflow Module

This module implements the complete command processing workflow that integrates
wake word detection with AI model communication for callback determination.

Features:
- Callback function registry for available commands
- AI model communication for command determination
- Callback execution and response generation
- Integration with wake word detection system
- Fallback mechanism for unrecognized commands
"""

import logging
import json
from typing import Dict, List, Optional, Callable, Any, Tuple
from dataclasses import dataclass, field
from datetime import datetime, timezone
import inspect

logger = logging.getLogger(__name__)


@dataclass
class CallbackFunction:
    """Represents a registered callback function"""
    name: str
    function: Callable
    description: str
    parameters: Dict[str, Any] = field(default_factory=dict)
    enabled: bool = True


@dataclass
class CommandProcessingResult:
    """Result of command processing workflow"""
    callback_executed: bool
    callback_name: Optional[str] = None
    callback_result: Any = None
    user_response: str = ""
    error: Optional[str] = None
    ai_response: Optional[Dict] = None


class CallbackRegistry:
    """Registry for managing available callback functions"""
    
    def __init__(self):
        self.callbacks: Dict[str, CallbackFunction] = {}
        self._setup_default_callbacks()
        logger.info("CallbackRegistry initialized")
    
    def _setup_default_callbacks(self):
        """Setup default callback functions"""
        # Register core callback functions
        self.register("getWeather", self._get_weather, 
                     "Get current weather information for a location")
        self.register("getTime", self._get_time, 
                     "Get current time")
        self.register("disableMira", self._disable_mira, 
                     "Disable the Mira assistant service")
    
    def register(self, name: str, function: Callable, description: str, 
                 parameters: Optional[Dict[str, Any]] = None) -> bool:
        """
        Register a new callback function
        
        Args:
            name: Function name to register
            function: The callable function
            description: Description of what the function does
            parameters: Optional parameter schema
            
        Returns:
            bool: True if registration successful
        """
        if not callable(function):
            logger.error(f"Cannot register non-callable object as callback: {name}")
            return False
        
        # Auto-extract parameters from function signature
        if parameters is None:
            try:
                sig = inspect.signature(function)
                parameters = {}
                for param_name, param in sig.parameters.items():
                    param_info = {
                        "type": param.annotation.__name__ if param.annotation != inspect.Parameter.empty else "str",
                        "required": param.default == inspect.Parameter.empty
                    }
                    parameters[param_name] = param_info
            except Exception as e:
                logger.warning(f"Could not extract parameters for {name}: {e}")
                parameters = {}
        
        callback = CallbackFunction(
            name=name,
            function=function,
            description=description,
            parameters=parameters
        )
        
        self.callbacks[name] = callback
        logger.info(f"Registered callback function: {name}")
        return True
    
    def unregister(self, name: str) -> bool:
        """
        Unregister a callback function
        
        Args:
            name: Function name to unregister
            
        Returns:
            bool: True if unregistration successful
        """
        if name in self.callbacks:
            del self.callbacks[name]
            logger.info(f"Unregistered callback function: {name}")
            return True
        
        logger.warning(f"Callback function not found: {name}")
        return False
    
    def execute(self, name: str, **kwargs) -> Tuple[bool, Any]:
        """
        Execute a registered callback function
        
        Args:
            name: Function name to execute
            **kwargs: Arguments to pass to the function
            
        Returns:
            Tuple[bool, Any]: (success, result)
        """
        if name not in self.callbacks:
            logger.error(f"Callback function not found: {name}")
            return False, f"Function '{name}' not found"
        
        callback = self.callbacks[name]
        if not callback.enabled:
            logger.warning(f"Callback function disabled: {name}")
            return False, f"Function '{name}' is disabled"
        
        try:
            result = callback.function(**kwargs)
            logger.info(f"Successfully executed callback: {name}")
            return True, result
        except Exception as e:
            logger.error(f"Error executing callback {name}: {e}")
            return False, str(e)
    
    def get_function_list(self) -> List[str]:
        """Get list of available function names"""
        return [name for name, callback in self.callbacks.items() if callback.enabled]
    
    def get_function_descriptions(self) -> Dict[str, str]:
        """Get mapping of function names to descriptions"""
        return {
            name: callback.description 
            for name, callback in self.callbacks.items() 
            if callback.enabled
        }
    
    def set_enabled(self, name: str, enabled: bool) -> bool:
        """Enable or disable a callback function"""
        if name in self.callbacks:
            self.callbacks[name].enabled = enabled
            status = "enabled" if enabled else "disabled"
            logger.info(f"Callback function {name} {status}")
            return True
        return False
    
    # Default callback implementations
    def _get_weather(self, location: str = "current location") -> str:
        """Get weather information (placeholder implementation)"""
        # This is a placeholder implementation
        # In a real system, this would integrate with a weather API
        return f"The weather in {location} is partly cloudy with a temperature of 72°F"
    
    def _get_time(self) -> str:
        """Get current time"""
        current_time = datetime.now().strftime("%I:%M %p")
        return f"The current time is {current_time}"
    
    def _disable_mira(self) -> str:
        """Disable the Mira assistant service"""
        # Import here to avoid circular imports
        try:
            from mira import status
            status["enabled"] = False
            return "Mira assistant has been disabled. Say 'Hey Mira' to re-enable."
        except ImportError:
            # Fallback if mira module isn't available
            return "Mira assistant has been disabled. Say 'Hey Mira' to re-enable."


class CommandProcessor:
    """Main command processing workflow orchestrator"""
    
    def __init__(self, callback_registry: Optional[CallbackRegistry] = None):
        """
        Initialize command processor
        
        Args:
            callback_registry: Optional callback registry, creates default if None
        """
        self.callback_registry = callback_registry or CallbackRegistry()
        logger.info("CommandProcessor initialized")
    
    def process_command(self, interaction_text: str, client_id: str, 
                       context: Optional[str] = None) -> CommandProcessingResult:
        """
        Process a command through the AI model and execute callbacks
        
        Args:
            interaction_text: The transcribed user interaction
            client_id: ID of the client that triggered the command
            context: Optional context information
            
        Returns:
            CommandProcessingResult: Result of command processing
        """
        logger.info(f"Processing command from client {client_id}: {interaction_text}")
        
        try:
            # Get AI response for callback determination
            ai_response = self._get_ai_callback_response(interaction_text, context)
            
            if not ai_response:
                return CommandProcessingResult(
                    callback_executed=False,
                    user_response="I didn't understand that command. Could you please try again?",
                    error="No AI response received"
                )
            
            # Extract callback information from AI response
            callback_name = ai_response.get("callback_function")
            callback_args = ai_response.get("callback_arguments", {})
            user_response = ai_response.get("user_response", "")
            
            # Execute callback if specified
            if callback_name:
                success, result = self.callback_registry.execute(callback_name, **callback_args)
                
                if success:
                    # Enhance user response with callback result if needed
                    if result and isinstance(result, str):
                        user_response = result if not user_response else f"{user_response} {result}"
                    
                    return CommandProcessingResult(
                        callback_executed=True,
                        callback_name=callback_name,
                        callback_result=result,
                        user_response=user_response,
                        ai_response=ai_response
                    )
                else:
                    return CommandProcessingResult(
                        callback_executed=False,
                        user_response=f"Sorry, I couldn't execute that command: {result}",
                        error=str(result),
                        ai_response=ai_response
                    )
            else:
                # No callback needed, just return AI response
                return CommandProcessingResult(
                    callback_executed=False,
                    user_response=user_response or "I understand, but no action is needed right now.",
                    ai_response=ai_response
                )
                
        except Exception as e:
            logger.error(f"Error processing command: {e}")
            return CommandProcessingResult(
                callback_executed=False,
                user_response="Sorry, I encountered an error processing that command.",
                error=str(e)
            )
    
    def _get_ai_callback_response(self, interaction_text: str, 
                                context: Optional[str] = None) -> Optional[Dict]:
        """
        Get AI model response for callback determination
        
        Args:
            interaction_text: User interaction text
            context: Optional context
            
        Returns:
            Dict: AI response with callback information
        """
        # Import inference processor here to avoid circular imports
        import inference_processor
        
        # Create enhanced prompt with available functions
        function_list = self.callback_registry.get_function_list()
        function_descriptions = self.callback_registry.get_function_descriptions()
        
        system_prompt = self._build_system_prompt(function_list, function_descriptions)
        
        # Build the complete prompt
        enhanced_prompt = f"""System: {system_prompt}

User Input: {interaction_text}"""
        
        if context:
            enhanced_prompt += f"\n\nContext: {context}"
        
        try:
            # Use existing inference processor to communicate with AI model
            ai_response = inference_processor.send_prompt(enhanced_prompt)
            return ai_response
        except Exception as e:
            logger.error(f"Error communicating with AI model: {e}")
            return None
    
    def _build_system_prompt(self, function_list: List[str], 
                           function_descriptions: Dict[str, str]) -> str:
        """
        Build system prompt with available functions
        
        Args:
            function_list: List of available function names
            function_descriptions: Function name to description mapping
            
        Returns:
            str: Complete system prompt
        """
        prompt = """You are Mira, an AI assistant that processes voice commands and determines which callback function (if any) should be invoked.

Available Functions:"""
        
        for func_name in function_list:
            description = function_descriptions.get(func_name, "No description available")
            prompt += f"\n- {func_name}: {description}"
        
        prompt += """

Your task:
1. Analyze the user input to determine if a callback function should be invoked
2. If a callback is needed, identify which function and any required arguments
3. Provide a user-facing response sentence

Respond with JSON in this format:
{
    "callback_function": "function_name" or null,
    "callback_arguments": {"arg1": "value1", "arg2": "value2"} or {},
    "user_response": "What to say back to the user"
}

Examples:
- "What time is it?" -> {"callback_function": "getTime", "callback_arguments": {}, "user_response": ""}
- "What's the weather like?" -> {"callback_function": "getWeather", "callback_arguments": {"location": "current location"}, "user_response": ""}
- "Stop listening" -> {"callback_function": "disableMira", "callback_arguments": {}, "user_response": ""}
- "Hello there" -> {"callback_function": null, "callback_arguments": {}, "user_response": "Hello! How can I help you?"}

Only invoke callbacks for clear, actionable requests. Respond naturally and conversationally."""
        
        return prompt


# Global instance for easy access
_command_processor: Optional[CommandProcessor] = None


def get_command_processor() -> CommandProcessor:
    """Get the global command processor instance"""
    global _command_processor
    if _command_processor is None:
        _command_processor = CommandProcessor()
    return _command_processor


def process_wake_word_command(interaction_text: str, client_id: str, 
                            context: Optional[str] = None) -> CommandProcessingResult:
    """
    Convenience function to process wake word triggered commands
    
    Args:
        interaction_text: The transcribed user interaction
        client_id: ID of the client that triggered the command
        context: Optional context information
        
    Returns:
        CommandProcessingResult: Result of command processing
    """
    processor = get_command_processor()
    return processor.process_command(interaction_text, client_id, context)