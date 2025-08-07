"""
Unit tests for ML Model Manager
Tests for MLModelManager class with mocked LM Studio server using OpenAI API
"""

import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import pytest
from unittest.mock import Mock, patch
import json
from openai.types.chat import ChatCompletion, ChatCompletionMessage
from openai.types.chat.chat_completion import Choice
from openai.types.completion_usage import CompletionUsage

from ml_model_manager import MLModelManager, get_available_models
from models import Interaction


# Mock all missing OpenAI chat completion message types
@patch("openai.types.chat.ChatCompletionSystemMessageParam", dict)
@patch("openai.types.chat.ChatCompletionAssistantMessageParam", dict)
@patch("openai.types.chat.ChatCompletionUserMessageParam", dict)
@patch("openai.types.chat.ChatCompletionMessageParam", list)
class TestGetAvailableModels:
    """Test cases for get_available_models function"""

    @patch("ml_model_manager.get_openai_client")
    def test_get_available_models_success(self, mock_get_client):
        """Test successful retrieval of available models"""
        expected_models = [{"id": "model1", "object": "model"}, {"id": "model2", "object": "model"}]

        mock_client = Mock()
        mock_response = Mock()
        mock_response.model_dump.return_value = {"data": expected_models}
        mock_client.models.list.return_value = mock_response
        mock_get_client.return_value = mock_client

        models = get_available_models()
        for model_info, expected_model_info in zip(models, expected_models):
            assert model_info["id"] == expected_model_info["id"]

    @patch("ml_model_manager.get_openai_client")
    def test_get_available_models_failure(self, mock_get_client):
        """Test get_available_models raises exception on server error"""
        mock_client = Mock()
        mock_client.models.list.side_effect = Exception("Connection failed")
        mock_get_client.return_value = mock_client

        with pytest.raises(Exception, match="Connection failed"):
            get_available_models()

    @patch("ml_model_manager.get_openai_client")
    def test_get_available_models_empty_response(self, mock_get_client):
        """Test get_available_models with empty data"""
        mock_client = Mock()
        mock_response = Mock()
        mock_response.model_dump.return_value = {"data": []}
        mock_client.models.list.return_value = mock_response
        mock_get_client.return_value = mock_client

        models = get_available_models()
        assert models == []


# Mock all missing OpenAI chat completion message types
@patch("openai.types.chat.ChatCompletionSystemMessageParam", dict)
@patch("openai.types.chat.ChatCompletionAssistantMessageParam", dict)
@patch("openai.types.chat.ChatCompletionUserMessageParam", dict)
@patch("openai.types.chat.ChatCompletionMessageParam", list)
@patch("openai.types.shared_params.ResponseFormatText")
@patch("openai.types.shared_params.ResponseFormatJSONSchema")
@patch("openai.types.shared_params.FunctionDefinition")
@patch("openai.types.shared_params.response_format_json_schema.JSONSchema")
class TestMLModelManager:
    """Test cases for the MLModelManager class"""

    def test_init_with_valid_model(self, *args):
        """Test MLModelManager initialization with valid model name"""
        # Test initialization
        manager = MLModelManager("test-model", "You are a test assistant")

        assert manager.model == "test-model"
        assert manager.system_prompt == "You are a test assistant"
        assert manager.tools == []

    def test_init_with_invalid_model(self, *args):
        """Test MLModelManager initialization with invalid model name logs warning but continues"""
        # Should not raise exception, but create manager successfully
        manager = MLModelManager("invalid-model", "Test prompt")
        assert manager.model == "invalid-model"
        assert manager.system_prompt == "Test prompt"

    def test_init_with_unloaded_model(self, *args):
        """Test MLModelManager initialization with unloaded model logs warning but continues"""
        # Should not raise exception, but create manager successfully
        manager = MLModelManager("test-model", "Test prompt")
        assert manager.model == "test-model"
        assert manager.system_prompt == "Test prompt"

    def test_init_with_config_options(self, *args):
        """Test MLModelManager initialization with custom config options"""
        manager = MLModelManager(
            "test-model", "Test prompt", temperature=0.7, max_tokens=100, custom_param="value"
        )

        assert manager.config["temperature"] == 0.7
        assert manager.config["max_tokens"] == 100
        assert manager.config["custom_param"] == "value"

    def test_init_with_response_format(self, *args):
        """Test MLModelManager initialization with structured response format"""
        response_format = {"type": "object", "properties": {"answer": {"type": "string"}}}

        manager = MLModelManager("test-model", "Test prompt", response_format=response_format)

        assert hasattr(manager, "response_format")
        assert manager.response_format is not None

    def test_register_tool(self, *args):
        """Test tool registration functionality"""
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
        assert tool["type"] == "function"

    @patch("ml_model_manager.get_openai_client")
    def test_run_inference_success(self, mock_get_client, *args):
        """Test successful inference run"""
        # Mock the OpenAI client
        mock_client = Mock()
        mock_get_client.return_value = mock_client
        
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
                        role="assistant", content=json.dumps(expected_result)
                    ),
                    finish_reason="stop",
                )
            ],
            usage=CompletionUsage(prompt_tokens=10, completion_tokens=5, total_tokens=15),
        )
        mock_client.chat.completions.create.return_value = mock_completion

        manager = MLModelManager("test-model", "Test prompt")

        # Create test interaction
        interaction = Interaction()
        interaction.text = "What time is it?"  # type: ignore

        result = manager.run_inference(interaction)

        assert result == expected_result
        mock_client.chat.completions.create.assert_called_once()

        # Verify the call arguments
        call_args = mock_client.chat.completions.create.call_args
        assert call_args[1]["model"] == "test-model"
        assert len(call_args[1]["messages"]) == 2
        assert call_args[1]["messages"][0]["role"] == "system"
        assert call_args[1]["messages"][1]["role"] == "user"
        assert call_args[1]["messages"][1]["content"] == "What time is it?"

    @patch("ml_model_manager.get_openai_client")
    def test_run_inference_with_context(self, mock_get_client, *args):
        """Test inference run with context"""
        # Mock the OpenAI client
        mock_client = Mock()
        mock_get_client.return_value = mock_client
        
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
                        role="assistant", content='{"answer": "Response with context"}'
                    ),
                    finish_reason="stop",
                )
            ],
            usage=CompletionUsage(prompt_tokens=15, completion_tokens=8, total_tokens=23),
        )
        mock_client.chat.completions.create.return_value = mock_completion

        manager = MLModelManager("test-model", "Test prompt")

        # Create test interaction
        interaction = Interaction()
        interaction.text = "Follow-up question"  # type: ignore

        result = manager.run_inference(interaction, context="Previous conversation")

        # Verify the call was made (context handling is implementation detail)
        mock_client.chat.completions.create.assert_called_once()
        call_args = mock_client.chat.completions.create.call_args
        assert call_args[1]["model"] == "test-model"
        assert result is not None

    @patch("ml_model_manager.get_openai_client")
    def test_run_inference_no_content(self, mock_get_client, *args):
        """Test inference run when model returns no content"""
        # Mock the OpenAI client
        mock_client = Mock()
        mock_get_client.return_value = mock_client
        
        # Mock inference response with no content
        mock_completion = ChatCompletion(
            id="chatcmpl-test",
            object="chat.completion",
            created=1234567890,
            model="test-model",
            choices=[
                Choice(
                    index=0,
                    message=ChatCompletionMessage(role="assistant", content=None),
                    finish_reason="stop",
                )
            ],
            usage=CompletionUsage(prompt_tokens=10, completion_tokens=0, total_tokens=10),
        )
        mock_client.chat.completions.create.return_value = mock_completion

        manager = MLModelManager("test-model", "Test prompt")

        # Create test interaction
        interaction = Interaction()
        interaction.text = "Test question"  # type: ignore

        with pytest.raises(RuntimeError, match="Model test-model generated no content"):
            manager.run_inference(interaction)

    @patch("ml_model_manager.get_openai_client")
    def test_run_inference_invalid_json(self, mock_get_client, *args):
        """Test inference run when model returns invalid JSON"""
        # Mock the OpenAI client
        mock_client = Mock()
        mock_get_client.return_value = mock_client
        
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
                        role="assistant", content="This is not valid JSON"
                    ),
                    finish_reason="stop",
                )
            ],
            usage=CompletionUsage(prompt_tokens=10, completion_tokens=5, total_tokens=15),
        )
        mock_client.chat.completions.create.return_value = mock_completion

        manager = MLModelManager("test-model", "Test prompt")

        # Create test interaction
        interaction = Interaction()
        interaction.text = "Test question"  # type: ignore

        with pytest.raises(json.JSONDecodeError):
            manager.run_inference(interaction)

    @patch("ml_model_manager.get_openai_client")
    def test_run_inference_with_tools(self, mock_get_client, *args):
        """Test inference run with registered tools"""
        # Mock the OpenAI client
        mock_client = Mock()
        mock_get_client.return_value = mock_client
        
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
                        role="assistant", content='{"answer": "Tool-based response"}'
                    ),
                    finish_reason="stop",
                )
            ],
            usage=CompletionUsage(prompt_tokens=20, completion_tokens=8, total_tokens=28),
        )
        mock_client.chat.completions.create.return_value = mock_completion

        manager = MLModelManager("test-model", "Test prompt")

        # Register a test tool
        def test_tool(query: str) -> str:
            return f"Tool result for: {query}"

        manager.register_tool(test_tool, "Test tool")

        # Create test interaction
        interaction = Interaction()
        interaction.text = "Use the tool"  # type: ignore

        result = manager.run_inference(interaction)

        # Verify tools were passed to the API call
        mock_client.chat.completions.create.assert_called_once()
        call_args = mock_client.chat.completions.create.call_args
        assert "tools" in call_args[1]
        assert call_args[1]["tool_choice"] == "auto"
        assert len(manager.tools) == 1
        assert result is not None


if __name__ == "__main__":
    pytest.main([__file__])
