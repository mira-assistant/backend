"""
Unit tests for ML Model Manager
Tests for MLModel and MLModelManager classes with mocked LM Studio server
"""

import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import pytest
from unittest.mock import Mock, patch, MagicMock
import json
import requests

from ml_model_manager import MLModel, MLModelManager, ModelResponse


class TestMLModel:
    """Test cases for the MLModel class"""
    
    @patch('ml_model_manager.requests.get')
    def test_init_with_valid_model(self, mock_get):
        """Test MLModel initialization with valid model name"""
        # Mock available models response
        mock_response = Mock()
        mock_response.json.return_value = {
            "data": [
                {"id": "microsoft/DialoGPT-small", "object": "model"},
                {"id": "gpt-3.5-turbo", "object": "model"}
            ]
        }
        mock_response.raise_for_status.return_value = None
        mock_get.return_value = mock_response
        
        # Test initialization
        model = MLModel("microsoft/DialoGPT-small")
        
        assert model.model == "microsoft/DialoGPT-small"
        assert model.system_prompt == "You are Mira assistant"
        assert model.config["temperature"] == 0.3
        assert model.config["max_tokens"] == -1
    
    @patch('ml_model_manager.requests.get')
    def test_init_with_custom_system_prompt(self, mock_get):
        """Test MLModel initialization with custom system prompt"""
        # Mock available models
        mock_response = Mock()
        mock_response.json.return_value = {
            "data": [{"id": "test-model", "object": "model"}]
        }
        mock_response.raise_for_status.return_value = None
        mock_get.return_value = mock_response
        
        custom_prompt = "You are a specialized assistant"
        model = MLModel("test-model", system_prompt=custom_prompt)
        
        assert model.system_prompt == custom_prompt
    
    @patch('ml_model_manager.requests.get')
    def test_init_with_config_options(self, mock_get):
        """Test MLModel initialization with custom config options"""
        # Mock available models
        mock_response = Mock()
        mock_response.json.return_value = {
            "data": [{"id": "test-model", "object": "model"}]
        }
        mock_response.raise_for_status.return_value = None
        mock_get.return_value = mock_response
        
        model = MLModel("test-model", temperature=0.7, max_tokens=100, custom_param="value")
        
        assert model.config["temperature"] == 0.7
        assert model.config["max_tokens"] == 100
        assert model.config["custom_param"] == "value"
        # Should preserve defaults for unspecified params
        assert model.config["top_p"] == 0.8
    
    @patch('ml_model_manager.requests.get')
    def test_init_with_invalid_model(self, mock_get):
        """Test MLModel initialization with invalid model name raises exception"""
        # Mock available models response
        mock_response = Mock()
        mock_response.json.return_value = {
            "data": [{"id": "valid-model", "object": "model"}]
        }
        mock_response.raise_for_status.return_value = None
        mock_get.return_value = mock_response
        
        with pytest.raises(ValueError, match="Model 'invalid-model' not found"):
            MLModel("invalid-model")
    
    @patch('ml_model_manager.requests.get')
    def test_get_available_models_success(self, mock_get):
        """Test successful retrieval of available models"""
        expected_models = [
            {"id": "model1", "object": "model"},
            {"id": "model2", "object": "model"}
        ]
        
        mock_response = Mock()
        mock_response.json.return_value = {"data": expected_models}
        mock_response.raise_for_status.return_value = None
        mock_get.return_value = mock_response
        
        models = MLModel.get_available_models()
        
        assert models == expected_models
        mock_get.assert_called_once_with("http://localhost:1234/v1/models")
    
    @patch('ml_model_manager.requests.get')
    def test_get_available_models_failure(self, mock_get):
        """Test get_available_models raises exception on server error"""
        mock_get.side_effect = requests.RequestException("Connection failed")
        
        with pytest.raises(requests.RequestException):
            MLModel.get_available_models()
    
    @patch('ml_model_manager.requests.get')
    @patch('ml_model_manager.requests.post')
    def test_run_inference_success(self, mock_post, mock_get):
        """Test successful inference run"""
        # Mock available models
        mock_get_response = Mock()
        mock_get_response.json.return_value = {
            "data": [{"id": "test-model", "object": "model"}]
        }
        mock_get_response.raise_for_status.return_value = None
        mock_get.return_value = mock_get_response
        
        # Mock inference response
        expected_result = {"callback_function": "getTime", "user_response": "Current time"}
        mock_post_response = Mock()
        mock_post_response.json.return_value = {
            "choices": [{"message": {"content": json.dumps(expected_result)}}]
        }
        mock_post_response.raise_for_status.return_value = None
        mock_post.return_value = mock_post_response
        
        model = MLModel("test-model")
        result = model.run_inference("What time is it?")
        
        assert result == expected_result
        mock_post.assert_called_once()
        
        # Verify payload structure
        call_args = mock_post.call_args
        payload = call_args[1]['json']
        assert payload['model'] == "test-model"
        assert len(payload['messages']) == 2
        assert payload['messages'][0]['role'] == 'system'
        assert payload['messages'][1]['role'] == 'user'
    
    @patch('ml_model_manager.requests.get')
    @patch('ml_model_manager.requests.post')
    def test_run_inference_with_context(self, mock_post, mock_get):
        """Test inference run with context"""
        # Mock available models
        mock_get_response = Mock()
        mock_get_response.json.return_value = {
            "data": [{"id": "test-model", "object": "model"}]
        }
        mock_get_response.raise_for_status.return_value = None
        mock_get.return_value = mock_get_response
        
        # Mock inference response
        mock_post_response = Mock()
        mock_post_response.json.return_value = {
            "choices": [{"message": {"content": '{"user_response": "Response with context"}'}}]
        }
        mock_post_response.raise_for_status.return_value = None
        mock_post.return_value = mock_post_response
        
        model = MLModel("test-model")
        result = model.run_inference("Follow-up question", context="Previous conversation")
        
        # Verify context was included in user prompt
        call_args = mock_post.call_args
        payload = call_args[1]['json']
        user_message = payload['messages'][1]['content']
        assert "Follow-up question" in user_message
        assert "Previous conversation" in user_message
    
    @patch('ml_model_manager.requests.get')
    @patch('ml_model_manager.requests.post')
    def test_run_inference_server_error(self, mock_post, mock_get):
        """Test inference run with server error"""
        # Mock available models
        mock_get_response = Mock()
        mock_get_response.json.return_value = {
            "data": [{"id": "test-model", "object": "model"}]
        }
        mock_get_response.raise_for_status.return_value = None
        mock_get.return_value = mock_get_response
        
        # Mock server error
        mock_post.side_effect = requests.RequestException("Server unavailable")
        
        model = MLModel("test-model")
        
        with pytest.raises(requests.RequestException):
            model.run_inference("Test question")


class TestMLModelManager:
    """Test cases for the MLModelManager class"""
    
    @patch('ml_model_manager.requests.get')
    def test_init_success(self, mock_get):
        """Test successful MLModelManager initialization"""
        # Mock available models
        mock_response = Mock()
        mock_response.json.return_value = {
            "data": [{"id": "test-model", "object": "model"}]
        }
        mock_response.raise_for_status.return_value = None
        mock_get.return_value = mock_response
        
        manager = MLModelManager("test-model")
        
        assert manager.command_model.model == "test-model"
        assert manager.action_model.model == "test-model"
        assert "Mira" in manager.command_model.system_prompt
        assert "action data" in manager.action_model.system_prompt
    
    @patch('ml_model_manager.requests.get')
    def test_init_with_custom_prompts(self, mock_get):
        """Test MLModelManager initialization with custom system prompts"""
        # Mock available models
        mock_response = Mock()
        mock_response.json.return_value = {
            "data": [{"id": "test-model", "object": "model"}]
        }
        mock_response.raise_for_status.return_value = None
        mock_get.return_value = mock_response
        
        command_prompt = "Custom command prompt"
        action_prompt = "Custom action prompt"
        
        manager = MLModelManager(
            "test-model",
            command_system_prompt=command_prompt,
            action_system_prompt=action_prompt
        )
        
        assert manager.command_model.system_prompt == command_prompt
        assert manager.action_model.system_prompt == action_prompt
    
    @patch('ml_model_manager.requests.get')
    def test_get_available_models(self, mock_get):
        """Test get_available_models static method"""
        expected_models = [{"id": "model1", "object": "model"}]
        
        mock_response = Mock()
        mock_response.json.return_value = {"data": expected_models}
        mock_response.raise_for_status.return_value = None
        mock_get.return_value = mock_response
        
        models = MLModelManager.get_available_models()
        
        assert models == expected_models
    
    @patch('ml_model_manager.requests.get')
    @patch('ml_model_manager.requests.post')
    def test_process_command_inference_success(self, mock_post, mock_get):
        """Test successful command inference processing"""
        # Mock available models
        mock_get_response = Mock()
        mock_get_response.json.return_value = {
            "data": [{"id": "test-model", "object": "model"}]
        }
        mock_get_response.raise_for_status.return_value = None
        mock_get.return_value = mock_get_response
        
        # Mock inference response
        expected_response = {
            "callback_function": "getTime",
            "callback_arguments": {},
            "user_response": "Let me get the current time for you"
        }
        mock_post_response = Mock()
        mock_post_response.json.return_value = {
            "choices": [{"message": {"content": json.dumps(expected_response)}}]
        }
        mock_post_response.raise_for_status.return_value = None
        mock_post.return_value = mock_post_response
        
        manager = MLModelManager("test-model")
        
        result = manager.process_command_inference(
            "What time is it?",
            ["getTime", "getWeather"],
            {"getTime": "Gets current time", "getWeather": "Gets weather info"}
        )
        
        assert isinstance(result, ModelResponse)
        assert result.success is True
        assert result.callback_function == "getTime"
        assert result.callback_arguments == {}
        assert result.user_response == "Let me get the current time for you"
    
    @patch('ml_model_manager.requests.get')
    @patch('ml_model_manager.requests.post')
    def test_process_command_inference_with_context(self, mock_post, mock_get):
        """Test command inference with context"""
        # Mock setup
        mock_get_response = Mock()
        mock_get_response.json.return_value = {
            "data": [{"id": "test-model", "object": "model"}]
        }
        mock_get_response.raise_for_status.return_value = None
        mock_get.return_value = mock_get_response
        
        mock_post_response = Mock()
        mock_post_response.json.return_value = {
            "choices": [{"message": {"content": '{"callback_function": null, "user_response": "Response"}'}}]
        }
        mock_post_response.raise_for_status.return_value = None
        mock_post.return_value = mock_post_response
        
        manager = MLModelManager("test-model")
        
        result = manager.process_command_inference(
            "Follow up question",
            ["testFunc"],
            {"testFunc": "Test function"},
            context="Previous context"
        )
        
        assert result.success is True
        
        # Verify context was passed to the model
        call_args = mock_post.call_args
        payload = call_args[1]['json']
        user_message = payload['messages'][1]['content']
        assert "Previous context" in user_message
    
    @patch('ml_model_manager.requests.get')
    @patch('ml_model_manager.requests.post')
    def test_process_action_extraction_success(self, mock_post, mock_get):
        """Test successful action extraction processing"""
        # Mock setup
        mock_get_response = Mock()
        mock_get_response.json.return_value = {
            "data": [{"id": "test-model", "object": "model"}]
        }
        mock_get_response.raise_for_status.return_value = None
        mock_get.return_value = mock_get_response
        
        expected_response = {
            "action_type": "calendar_entry",
            "action_data": {
                "title": "Meeting with John",
                "start_time": "2024-01-15T14:00:00Z",
                "description": "Quarterly review meeting"
            },
            "user_response": "I've scheduled your meeting with John"
        }
        mock_post_response = Mock()
        mock_post_response.json.return_value = {
            "choices": [{"message": {"content": json.dumps(expected_response)}}]
        }
        mock_post_response.raise_for_status.return_value = None
        mock_post.return_value = mock_post_response
        
        manager = MLModelManager("test-model")
        
        result = manager.process_action_extraction(
            "Schedule a meeting with John tomorrow at 2 PM for quarterly review"
        )
        
        assert isinstance(result, ModelResponse)
        assert result.success is True
        assert result.action_type == "calendar_entry"
        assert result.action_data["title"] == "Meeting with John"
        assert result.user_response == "I've scheduled your meeting with John"
    
    @patch('ml_model_manager.requests.get')
    @patch('ml_model_manager.requests.post')
    def test_process_recursive_command_single_iteration(self, mock_post, mock_get):
        """Test recursive command processing with single iteration"""
        # Mock setup
        mock_get_response = Mock()
        mock_get_response.json.return_value = {
            "data": [{"id": "test-model", "object": "model"}]
        }
        mock_get_response.raise_for_status.return_value = None
        mock_get.return_value = mock_get_response
        
        # Single callback response
        response_data = {
            "callback_function": "getTime",
            "callback_arguments": {},
            "user_response": "The current time is 2:30 PM"
        }
        mock_post_response = Mock()
        mock_post_response.json.return_value = {
            "choices": [{"message": {"content": json.dumps(response_data)}}]
        }
        mock_post_response.raise_for_status.return_value = None
        mock_post.return_value = mock_post_response
        
        manager = MLModelManager("test-model")
        
        # Mock callback executor that returns empty string to stop recursion
        callback_executor = Mock()
        callback_executor.return_value = ""
        
        result = manager.process_recursive_command(
            "What time is it?",
            ["getTime"],
            {"getTime": "Gets current time"},
            callback_executor
        )
        
        assert result.success is True
        assert result.callback_function == "getTime"
        callback_executor.assert_called_once_with("getTime", {})
    
    @patch('ml_model_manager.requests.get') 
    @patch('ml_model_manager.requests.post')
    def test_process_recursive_command_multiple_iterations(self, mock_post, mock_get):
        """Test recursive command processing with multiple iterations"""
        # Mock setup
        mock_get_response = Mock()
        mock_get_response.json.return_value = {
            "data": [{"id": "test-model", "object": "model"}]
        }
        mock_get_response.raise_for_status.return_value = None
        mock_get.return_value = mock_get_response
        
        # Set up responses for multiple iterations
        responses = [
            # First call - triggers callback
            {
                "callback_function": "getWeather", 
                "callback_arguments": {"location": "current"},
                "user_response": "Let me get the weather"
            },
            # Second call - triggered by callback result, no more callbacks
            {
                "callback_function": None,
                "callback_arguments": {},
                "user_response": "The weather is sunny, 72°F"
            }
        ]
        
        mock_post_response = Mock()
        # Return different responses for each call
        mock_post_response.json.side_effect = [
            {"choices": [{"message": {"content": json.dumps(responses[0])}}]},
            {"choices": [{"message": {"content": json.dumps(responses[1])}}]}
        ]
        mock_post_response.raise_for_status.return_value = None
        mock_post.return_value = mock_post_response
        
        manager = MLModelManager("test-model")
        
        # Mock callback executor that returns string to trigger second iteration
        callback_executor = Mock()
        callback_executor.return_value = "get detailed weather forecast"
        
        result = manager.process_recursive_command(
            "What's the weather like?",
            ["getWeather"],
            {"getWeather": "Gets weather information"},
            callback_executor
        )
        
        assert result.success is True
        assert result.callback_function is None  # Final response has no callback
        assert result.user_response == "The weather is sunny, 72°F"
        callback_executor.assert_called_once_with("getWeather", {"location": "current"})
        
        # Verify two POST calls were made (for two iterations)
        assert mock_post.call_count == 2
    
    @patch('ml_model_manager.requests.get')
    @patch('ml_model_manager.requests.post')
    def test_process_recursive_command_no_callback(self, mock_post, mock_get):
        """Test recursive command processing with no callback"""
        # Mock setup
        mock_get_response = Mock()
        mock_get_response.json.return_value = {
            "data": [{"id": "test-model", "object": "model"}]
        }
        mock_get_response.raise_for_status.return_value = None
        mock_get.return_value = mock_get_response
        
        # No callback response
        response_data = {
            "callback_function": None,
            "callback_arguments": {},
            "user_response": "Hello! How can I help you?"
        }
        mock_post_response = Mock()
        mock_post_response.json.return_value = {
            "choices": [{"message": {"content": json.dumps(response_data)}}]
        }
        mock_post_response.raise_for_status.return_value = None
        mock_post.return_value = mock_post_response
        
        manager = MLModelManager("test-model")
        
        callback_executor = Mock()
        
        result = manager.process_recursive_command(
            "Hello there",
            ["getTime"],
            {"getTime": "Gets current time"},
            callback_executor
        )
        
        assert result.success is True
        assert result.callback_function is None
        callback_executor.assert_not_called()
    
    @patch('ml_model_manager.requests.get')
    def test_process_recursive_command_callback_failure(self, mock_get):
        """Test recursive command processing with callback execution failure"""
        # Mock setup
        mock_get_response = Mock()
        mock_get_response.json.return_value = {
            "data": [{"id": "test-model", "object": "model"}]
        }
        mock_get_response.raise_for_status.return_value = None
        mock_get.return_value = mock_get_response
        
        manager = MLModelManager("test-model")
        
        # Mock callback executor that raises exception
        callback_executor = Mock()
        callback_executor.side_effect = Exception("Callback failed")
        
        # Mock the command inference to return a callback
        with patch.object(manager, 'process_command_inference') as mock_inference:
            mock_response = ModelResponse(
                callback_function="failingCallback",
                callback_arguments={},
                user_response="This will fail",
                success=True
            )
            mock_inference.return_value = mock_response
            
            result = manager.process_recursive_command(
                "Test command",
                ["failingCallback"],
                {"failingCallback": "A callback that fails"},
                callback_executor
            )
            
            assert result.success is False
            assert "Callback execution failed" in result.error
            assert "error while processing" in result.user_response


class TestModelResponse:
    """Test cases for the ModelResponse dataclass"""
    
    def test_default_initialization(self):
        """Test ModelResponse initialization with defaults"""
        response = ModelResponse()
        
        assert response.callback_function is None
        assert response.callback_arguments == {}
        assert response.action_data is None
        assert response.action_type is None
        assert response.user_response == ""
        assert response.success is True
        assert response.error is None
    
    def test_custom_initialization(self):
        """Test ModelResponse initialization with custom values"""
        custom_args = {"param1": "value1", "param2": "value2"}
        action_data = {"title": "Test Event", "time": "2024-01-15T10:00:00Z"}
        
        response = ModelResponse(
            callback_function="testCallback",
            callback_arguments=custom_args,
            action_data=action_data,
            action_type="calendar_entry",
            user_response="Custom response",
            success=False,
            error="Test error"
        )
        
        assert response.callback_function == "testCallback"
        assert response.callback_arguments == custom_args
        assert response.action_data == action_data
        assert response.action_type == "calendar_entry"
        assert response.user_response == "Custom response"
        assert response.success is False
        assert response.error == "Test error"
    
    def test_post_init_callback_arguments(self):
        """Test that callback_arguments defaults to empty dict if None"""
        response = ModelResponse(callback_arguments=None)
        assert response.callback_arguments == {}
        
        # Test that explicit empty dict is preserved
        response2 = ModelResponse(callback_arguments={})
        assert response2.callback_arguments == {}
        
        # Test that existing dict is preserved
        custom_args = {"test": "value"}
        response3 = ModelResponse(callback_arguments=custom_args)
        assert response3.callback_arguments == custom_args


if __name__ == "__main__":
    pytest.main([__file__])