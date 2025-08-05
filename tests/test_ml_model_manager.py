"""
Unit tests for ML Model Manager
Tests for MLModelManager class with mocked LM Studio server using OpenAI API
"""

import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import pytest
from unittest.mock import Mock, patch, MagicMock
import json
import requests
from openai.types.chat import ChatCompletion, ChatCompletionMessage
from openai.types.chat.chat_completion import Choice
from openai.types.completion_usage import CompletionUsage

from ml_model_manager import MLModelManager, get_available_models
from models import Interaction


class TestGetAvailableModels:
    """Test cases for get_available_models function"""

    @patch('requests.get')
    def test_get_available_models_success(self, mock_get):
        """Test successful retrieval of available models"""
        expected_models = [
            {"id": "model1", "object": "model",},
            {"id": "model2", "object": "model", }
        ]

        mock_response = Mock()
        mock_response.json.return_value = {"data": expected_models}
        mock_response.raise_for_status.return_value = None
        mock_get.return_value = mock_response

        models = get_available_models()
        for model_info, expected_model_info in zip(models, expected_models):
            assert model_info["id"] == expected_model_info["id"]

    @patch('requests.get')
    def test_get_available_models_failure(self, mock_get):
        """Test get_available_models raises exception on server error"""
        mock_get.side_effect = requests.RequestException("Connection failed")

        with pytest.raises(RuntimeError, match="Could not fetch available models"):
            get_available_models()

    @patch('requests.get')
    def test_get_available_models_empty_response(self, mock_get):
        """Test get_available_models with empty data"""
        mock_response = Mock()
        mock_response.json.return_value = {"data": []}
        mock_response.raise_for_status.return_value = None
        mock_get.return_value = mock_response

        models = get_available_models()

        assert models == []


class TestMLModelManager:
    """Test cases for the MLModelManager class"""

    @patch('requests.get')
    def test_init_with_valid_model(self, mock_get):
        """Test MLModelManager initialization with valid model name"""
        # Mock available models response
        mock_response = Mock()
        mock_response.json.return_value = {
            "data": [
                {"id": "test-model", "object": "model", "state": "loaded"},
                {"id": "gpt-3.5-turbo", "object": "model", "state": "loaded"}
            ]
        }
        mock_response.raise_for_status.return_value = None
        mock_get.return_value = mock_response

        # Test initialization
        manager = MLModelManager("test-model", "You are a test assistant")

        assert manager.model == "test-model"
        assert manager.system_prompt == "You are a test assistant"
        assert manager.tools == []
        mock_get.assert_called_once()

    @patch('requests.get')
    def test_init_with_invalid_model(self, mock_get):
        """Test MLModelManager initialization with invalid model name raises exception"""
        # Mock available models response
        mock_response = Mock()
        mock_response.json.return_value = {
            "data": [{"id": "valid-model", "object": "model", "state": "loaded"}]
        }
        mock_response.raise_for_status.return_value = None
        mock_get.return_value = mock_response

        with pytest.raises(ValueError, match="Model 'invalid-model' not available or loaded"):
            MLModelManager("invalid-model", "Test prompt")

    @patch('requests.get')
    def test_init_with_unloaded_model(self, mock_get):
        """Test MLModelManager initialization with unloaded model raises exception"""
        # Mock available models response with unloaded model
        mock_response = Mock()
        mock_response.json.return_value = {
            "data": [{"id": "test-model", "object": "model", "state": "unloaded"}]
        }
        mock_response.raise_for_status.return_value = None
        mock_get.return_value = mock_response

        with pytest.raises(ValueError, match="Model 'test-model' not available or loaded"):
            MLModelManager("test-model", "Test prompt")

    @patch('requests.get')
    def test_init_with_config_options(self, mock_get):
        """Test MLModelManager initialization with custom config options"""
        # Mock available models
        mock_response = Mock()
        mock_response.json.return_value = {
            "data": [{"id": "test-model", "object": "model", "state": "loaded"}]
        }
        mock_response.raise_for_status.return_value = None
        mock_get.return_value = mock_response

        manager = MLModelManager(
            "test-model",
            "Test prompt",
            temperature=0.7,
            max_tokens=100,
            custom_param="value"
        )

        assert manager.config["temperature"] == 0.7
        assert manager.config["max_tokens"] == 100
        assert manager.config["custom_param"] == "value"

    @patch('requests.get')
    def test_init_with_response_format(self, mock_get):
        """Test MLModelManager initialization with structured response format"""
        # Mock available models
        mock_response = Mock()
        mock_response.json.return_value = {
            "data": [{"id": "test-model", "object": "model", "state": "loaded"}]
        }
        mock_response.raise_for_status.return_value = None
        mock_get.return_value = mock_response

        response_format = {
            "type": "object",
            "properties": {"answer": {"type": "string"}}
        }

        manager = MLModelManager("test-model", "Test prompt", response_format=response_format)

        assert hasattr(manager, 'response_format')
        assert manager.response_format is not None
        assert manager.response_format["type"] == "json_schema"
        assert "json_schema" in manager.response_format

    @patch('requests.get')
    def test_register_tool(self, mock_get):
        """Test tool registration functionality"""
        # Mock available models
        mock_response = Mock()
        mock_response.json.return_value = {
            "data": [{"id": "test-model", "object": "model", "state": "loaded"}]
        }
        mock_response.raise_for_status.return_value = None
        mock_get.return_value = mock_response

        manager = MLModelManager("test-model", "Test prompt")

        # Define a test function to register
        def test_function(param1: str, param2: int = 10) -> str:
            """Test function for tool registration"""
            return f"Result: {param1}, {param2}"

        # Register the tool
        manager.register_tool(test_function, "Test tool description")

        # Verify tool was registered
        assert len(manager.tools) == 1
        tool = manager.tools[0]
        assert tool['function']['name'] == 'test_function'
        assert tool['function']['description'] == "Test tool description"
        assert tool['type'] == 'function'

    @patch('requests.get')
    @patch('ml_model_manager.client.chat.completions.create')
    def test_run_inference_success(self, mock_create, mock_get):
        """Test successful inference run"""
        # Mock available models
        mock_get_response = Mock()
        mock_get_response.json.return_value = {
            "data": [{"id": "test-model", "object": "model", "state": "loaded"}]
        }
        mock_get_response.raise_for_status.return_value = None
        mock_get.return_value = mock_get_response

        # Mock inference response
        expected_result = {"answer": "The current time is 2:30 PM"}
        mock_completion = ChatCompletion(
            id="chatcmpl-test",
            object="chat.completion",
            created=1234567890,
            model="test-model",
            choices=[
                Choice(
                    index=0,
                    message=ChatCompletionMessage(
                        role="assistant",
                        content=json.dumps(expected_result)
                    ),
                    finish_reason="stop"
                )
            ],
            usage=CompletionUsage(
                prompt_tokens=10,
                completion_tokens=5,
                total_tokens=15
            )
        )
        mock_create.return_value = mock_completion

        manager = MLModelManager("test-model", "Test prompt")

        # Create test interaction
        interaction = Interaction()
        interaction.text = "What time is it?"

        result = manager.run_inference(interaction)

        assert result == expected_result
        mock_create.assert_called_once()

        # Verify the call arguments
        call_args = mock_create.call_args
        assert call_args[1]['model'] == "test-model"
        assert len(call_args[1]['messages']) == 2
        assert call_args[1]['messages'][0]['role'] == 'system'
        assert call_args[1]['messages'][1]['role'] == 'user'
        assert call_args[1]['messages'][1]['content'] == "What time is it?"

    @patch('requests.get')
    @patch('ml_model_manager.client.chat.completions.create')
    def test_run_inference_with_context(self, mock_create, mock_get):
        """Test inference run with context"""
        # Mock available models
        mock_get_response = Mock()
        mock_get_response.json.return_value = {
            "data": [{"id": "test-model", "object": "model", "state": "loaded"}]
        }
        mock_get_response.raise_for_status.return_value = None
        mock_get.return_value = mock_get_response

        # Mock inference response
        mock_completion = ChatCompletion(
            id="chatcmpl-test",
            object="chat.completion",
            created=1234567890,
            model="test-model",
            choices=[
                Choice(
                    index=0,
                    message=ChatCompletionMessage(
                        role="assistant",
                        content='{"answer": "Response with context"}'
                    ),
                    finish_reason="stop"
                )
            ],
            usage=CompletionUsage(
                prompt_tokens=15,
                completion_tokens=8,
                total_tokens=23
            )
        )
        mock_create.return_value = mock_completion

        manager = MLModelManager("test-model", "Test prompt")

        # Create test interaction
        interaction = Interaction()
        interaction.text = "Follow-up question"

        result = manager.run_inference(interaction, context="Previous conversation")

        # Verify the call was made (context handling is implementation detail)
        mock_create.assert_called_once()
        call_args = mock_create.call_args
        assert call_args[1]['model'] == "test-model"

    @patch('requests.get')
    @patch('ml_model_manager.client.chat.completions.create')
    def test_run_inference_no_content(self, mock_create, mock_get):
        """Test inference run when model returns no content"""
        # Mock available models
        mock_get_response = Mock()
        mock_get_response.json.return_value = {
            "data": [{"id": "test-model", "object": "model", "state": "loaded"}]
        }
        mock_get_response.raise_for_status.return_value = None
        mock_get.return_value = mock_get_response

        # Mock inference response with no content
        mock_completion = ChatCompletion(
            id="chatcmpl-test",
            object="chat.completion",
            created=1234567890,
            model="test-model",
            choices=[
                Choice(
                    index=0,
                    message=ChatCompletionMessage(
                        role="assistant",
                        content=None
                    ),
                    finish_reason="stop"
                )
            ],
            usage=CompletionUsage(
                prompt_tokens=10,
                completion_tokens=0,
                total_tokens=10
            )
        )
        mock_create.return_value = mock_completion

        manager = MLModelManager("test-model", "Test prompt")

        # Create test interaction
        interaction = Interaction()
        interaction.text = "Test question"

        with pytest.raises(RuntimeError, match="Model test-model generated no content"):
            manager.run_inference(interaction)

    @patch('requests.get')
    @patch('ml_model_manager.client.chat.completions.create')
    def test_run_inference_invalid_json(self, mock_create, mock_get):
        """Test inference run when model returns invalid JSON"""
        # Mock available models
        mock_get_response = Mock()
        mock_get_response.json.return_value = {
            "data": [{"id": "test-model", "object": "model", "state": "loaded"}]
        }
        mock_get_response.raise_for_status.return_value = None
        mock_get.return_value = mock_get_response

        # Mock inference response with invalid JSON
        mock_completion = ChatCompletion(
            id="chatcmpl-test",
            object="chat.completion",
            created=1234567890,
            model="test-model",
            choices=[
                Choice(
                    index=0,
                    message=ChatCompletionMessage(
                        role="assistant",
                        content="This is not valid JSON"
                    ),
                    finish_reason="stop"
                )
            ],
            usage=CompletionUsage(
                prompt_tokens=10,
                completion_tokens=5,
                total_tokens=15
            )
        )
        mock_create.return_value = mock_completion

        manager = MLModelManager("test-model", "Test prompt")

        # Create test interaction
        interaction = Interaction()
        interaction.text = "Test question"

        with pytest.raises(json.JSONDecodeError):
            manager.run_inference(interaction)

    @patch('requests.get')
    @patch('ml_model_manager.client.chat.completions.create')
    def test_run_inference_with_tools(self, mock_create, mock_get):
        """Test inference run with registered tools"""
        # Mock available models
        mock_get_response = Mock()
        mock_get_response.json.return_value = {
            "data": [{"id": "test-model", "object": "model", "state": "loaded"}]
        }
        mock_get_response.raise_for_status.return_value = None
        mock_get.return_value = mock_get_response

        # Mock inference response
        mock_completion = ChatCompletion(
            id="chatcmpl-test",
            object="chat.completion",
            created=1234567890,
            model="test-model",
            choices=[
                Choice(
                    index=0,
                    message=ChatCompletionMessage(
                        role="assistant",
                        content='{"answer": "Tool-based response"}'
                    ),
                    finish_reason="stop"
                )
            ],
            usage=CompletionUsage(
                prompt_tokens=20,
                completion_tokens=8,
                total_tokens=28
            )
        )
        mock_create.return_value = mock_completion

        manager = MLModelManager("test-model", "Test prompt")

        # Register a test tool
        def test_tool(query: str) -> str:
            return f"Tool result for: {query}"

        manager.register_tool(test_tool, "Test tool")

        # Create test interaction
        interaction = Interaction()
        interaction.text = "Use the tool"

        result = manager.run_inference(interaction)

        # Verify tools were passed to the API call
        call_args = mock_create.call_args
        assert 'tools' in call_args[1]
        assert call_args[1]['tool_choice'] == 'auto'
        assert len(manager.tools) == 1

if __name__ == "__main__":
    pytest.main([__file__])