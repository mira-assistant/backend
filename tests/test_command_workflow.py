"""
Unit tests for CommandWorkflow and CallbackRegistry from command_workflow.py
"""

import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import pytest
from unittest.mock import Mock, patch, MagicMock

from command_workflow import (
    CallbackRegistry, 
    CommandProcessor, 
    CallbackFunction, 
    CommandProcessingResult,
    get_command_processor,
    process_wake_word_command
)


class TestCallbackRegistry:
    def setup_method(self):
        self.registry = CallbackRegistry()

    def test_initialization(self):
        """Test that registry initializes with default callbacks"""
        callbacks = self.registry.get_function_list()
        assert "getWeather" in callbacks
        assert "getTime" in callbacks
        assert "disableMira" in callbacks
        assert len(callbacks) >= 3

    def test_register_callback(self):
        """Test registering a new callback function"""
        def test_func(name: str = "default"):
            return f"Hello {name}"
        
        result = self.registry.register("testFunc", test_func, "Test function")
        assert result is True
        assert "testFunc" in self.registry.get_function_list()
        
        descriptions = self.registry.get_function_descriptions()
        assert descriptions["testFunc"] == "Test function"

    def test_register_non_callable(self):
        """Test that registering non-callable fails"""
        result = self.registry.register("invalid", "not a function", "Invalid")
        assert result is False
        assert "invalid" not in self.registry.get_function_list()

    def test_unregister_callback(self):
        """Test unregistering a callback function"""
        def test_func():
            return "test"
        
        self.registry.register("toRemove", test_func, "To be removed")
        assert "toRemove" in self.registry.get_function_list()
        
        result = self.registry.unregister("toRemove")
        assert result is True
        assert "toRemove" not in self.registry.get_function_list()

    def test_unregister_nonexistent(self):
        """Test unregistering non-existent callback"""
        result = self.registry.unregister("nonExistent")
        assert result is False

    def test_execute_callback(self):
        """Test executing a registered callback"""
        def test_func(message: str):
            return f"Executed: {message}"
        
        self.registry.register("execTest", test_func, "Execution test")
        success, result = self.registry.execute("execTest", message="Hello")
        
        assert success is True
        assert result == "Executed: Hello"

    def test_execute_nonexistent_callback(self):
        """Test executing non-existent callback"""
        success, result = self.registry.execute("nonExistent")
        assert success is False
        assert "not found" in result

    def test_execute_disabled_callback(self):
        """Test executing disabled callback"""
        def test_func():
            return "test"
        
        self.registry.register("disabledTest", test_func, "Disabled test")
        self.registry.set_enabled("disabledTest", False)
        
        success, result = self.registry.execute("disabledTest")
        assert success is False
        assert "disabled" in result

    def test_enable_disable_callback(self):
        """Test enabling and disabling callbacks"""
        def test_func():
            return "test"
        
        self.registry.register("toggleTest", test_func, "Toggle test")
        
        # Test disable
        result = self.registry.set_enabled("toggleTest", False)
        assert result is True
        assert "toggleTest" not in self.registry.get_function_list()
        
        # Test enable
        result = self.registry.set_enabled("toggleTest", True)
        assert result is True
        assert "toggleTest" in self.registry.get_function_list()

    def test_default_callbacks_execution(self):
        """Test that default callbacks can be executed"""
        # Test getTime
        success, result = self.registry.execute("getTime")
        assert success is True
        assert "time is" in result.lower()
        
        # Test getWeather
        success, result = self.registry.execute("getWeather", location="New York")
        assert success is True
        assert "weather" in result.lower()
        
        # Test disableMira
        success, result = self.registry.execute("disableMira")
        assert success is True
        assert "disabled" in result.lower()


class TestCommandProcessor:
    def setup_method(self):
        self.registry = CallbackRegistry()
        self.processor = CommandProcessor(self.registry)

    @patch('inference_processor.send_prompt')
    def test_process_command_with_callback(self, mock_send_prompt):
        """Test processing command that triggers a callback"""
        # Mock AI response
        mock_send_prompt.return_value = {
            "callback_function": "getTime",
            "callback_arguments": {},
            "user_response": "Here's the current time:"
        }
        
        result = self.processor.process_command("What time is it?", "test_client")
        
        assert result.callback_executed is True
        assert result.callback_name == "getTime"
        assert result.error is None
        assert "time" in result.user_response.lower()

    @patch('inference_processor.send_prompt')
    def test_process_command_without_callback(self, mock_send_prompt):
        """Test processing command that doesn't trigger a callback"""
        # Mock AI response
        mock_send_prompt.return_value = {
            "callback_function": None,
            "callback_arguments": {},
            "user_response": "Hello! How can I help you?"
        }
        
        result = self.processor.process_command("Hello", "test_client")
        
        assert result.callback_executed is False
        assert result.callback_name is None
        assert result.error is None
        assert "Hello" in result.user_response

    @patch('inference_processor.send_prompt')
    def test_process_command_callback_failure(self, mock_send_prompt):
        """Test processing command when callback execution fails"""
        # Mock AI response
        mock_send_prompt.return_value = {
            "callback_function": "nonExistentFunction",
            "callback_arguments": {},
            "user_response": "I'll help with that."
        }
        
        result = self.processor.process_command("Do something", "test_client")
        
        assert result.callback_executed is False
        assert result.error is not None
        assert "couldn't execute" in result.user_response

    @patch('inference_processor.send_prompt')
    def test_process_command_ai_error(self, mock_send_prompt):
        """Test processing command when AI communication fails"""
        mock_send_prompt.side_effect = Exception("AI communication error")
        
        result = self.processor.process_command("Test command", "test_client")
        
        assert result.callback_executed is False
        assert result.error == "No AI response received"
        assert "didn't understand" in result.user_response

    @patch('inference_processor.send_prompt')
    def test_process_command_no_ai_response(self, mock_send_prompt):
        """Test processing command when AI returns None"""
        mock_send_prompt.return_value = None
        
        result = self.processor.process_command("Test command", "test_client")
        
        assert result.callback_executed is False
        assert result.error == "No AI response received"
        assert "didn't understand" in result.user_response

    def test_build_system_prompt(self):
        """Test system prompt building"""
        function_list = ["getTime", "getWeather"]
        function_descriptions = {
            "getTime": "Get current time",
            "getWeather": "Get weather information"
        }
        
        prompt = self.processor._build_system_prompt(function_list, function_descriptions)
        
        assert "getTime" in prompt
        assert "getWeather" in prompt
        assert "Get current time" in prompt
        assert "Get weather information" in prompt
        assert "JSON" in prompt


class TestGlobalFunctions:
    def test_get_command_processor(self):
        """Test global command processor getter"""
        processor1 = get_command_processor()
        processor2 = get_command_processor()
        
        assert processor1 is processor2  # Should return same instance
        assert isinstance(processor1, CommandProcessor)

    @patch('command_workflow.get_command_processor')
    def test_process_wake_word_command(self, mock_get_processor):
        """Test convenience function for wake word command processing"""
        mock_processor = Mock()
        mock_result = CommandProcessingResult(
            callback_executed=True,
            callback_name="getTime",
            user_response="The time is 2:00 PM"
        )
        mock_processor.process_command.return_value = mock_result
        mock_get_processor.return_value = mock_processor
        
        result = process_wake_word_command("What time is it?", "test_client")
        
        assert result.callback_executed is True
        assert result.callback_name == "getTime"
        mock_processor.process_command.assert_called_once_with(
            "What time is it?", "test_client", None
        )


class TestCallbackFunction:
    def test_callback_function_creation(self):
        """Test CallbackFunction dataclass creation"""
        def test_func():
            return "test"
        
        callback = CallbackFunction(
            name="test",
            function=test_func,
            description="Test function"
        )
        
        assert callback.name == "test"
        assert callback.function == test_func
        assert callback.description == "Test function"
        assert callback.enabled is True
        assert callback.parameters == {}


class TestCommandProcessingResult:
    def test_result_creation(self):
        """Test CommandProcessingResult dataclass creation"""
        result = CommandProcessingResult(
            callback_executed=True,
            callback_name="getTime",
            user_response="Time response"
        )
        
        assert result.callback_executed is True
        assert result.callback_name == "getTime"
        assert result.user_response == "Time response"
        assert result.error is None


if __name__ == "__main__":
    pytest.main([__file__, "-v"])