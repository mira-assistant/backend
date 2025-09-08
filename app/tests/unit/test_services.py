"""
Unit tests for service classes.
"""

from unittest.mock import Mock, patch, mock_open
from app.services.service_factory import (
    ServiceFactory,
    get_command_processor,
    get_sentence_processor,
)
from app.services.service_registry import service_registry


# Helper functions for mocking
def mock_open_with_content(content):
    """Helper function to mock open with specific content."""
    return mock_open(read_data=content)


class TestServiceFactory:
    """Test cases for ServiceFactory class."""

    @patch.dict("os.environ", {"GEMINI_API_KEY": "test_key"})
    @patch("app.services.command_processor.genai.Client")
    @patch("app.services.service_factory.ServiceFactory._load_network_config")
    @patch("app.services.service_factory.ServiceFactory._load_default_system_prompt")
    def test_create_command_processor(self, mock_load_prompt, mock_load_config, mock_genai_client):
        """Test creating a CommandProcessor with direct Gemini integration."""
        # Setup mocks
        mock_config = {"system_prompt": "test prompt"}
        mock_load_config.return_value = mock_config
        mock_load_prompt.return_value = "test prompt"
        mock_genai_client.return_value = Mock()

        # Test
        processor = ServiceFactory.create_command_processor("test-network")

        # Assertions
        mock_load_config.assert_called_once_with("test-network")
        mock_load_prompt.assert_called_once()
        assert processor is not None
        assert processor.network_id == "test-network"
        assert processor.system_prompt == "test prompt"

    @patch("app.services.service_factory.ServiceFactory._load_network_config")
    @patch("app.services.context_processor.ContextProcessor")
    def test_create_context_processor(self, mock_processor_class, mock_load_config):
        """Test creating a ContextProcessor."""
        # Setup mocks
        mock_config = {"test": "config"}
        mock_load_config.return_value = mock_config
        mock_processor_instance = Mock()
        mock_processor_class.return_value = mock_processor_instance

        # Test
        processor = ServiceFactory.create_context_processor("test-network")

        # Assertions
        mock_load_config.assert_called_once_with("test-network")
        mock_processor_class.assert_called_once_with("test-network", mock_config)
        assert processor == mock_processor_instance

    @patch.dict("os.environ", {"GEMINI_API_KEY": "test_key"})
    @patch("app.services.inference_processor.genai.Client")
    @patch("app.services.service_factory.ServiceFactory._load_network_config")
    @patch("app.services.service_factory.ServiceFactory._load_action_system_prompt")
    @patch("app.services.service_factory.ServiceFactory._load_action_response_format")
    def test_create_inference_processor(
        self, mock_load_format, mock_load_prompt, mock_load_config, mock_genai_client
    ):
        """Test creating an InferenceProcessor with direct Gemini integration."""
        # Setup mocks
        mock_config = {"system_prompt": "test prompt", "response_format": {"test": "format"}}
        mock_load_config.return_value = mock_config
        mock_load_prompt.return_value = "test prompt"
        mock_load_format.return_value = {"test": "format"}
        mock_genai_client.return_value = Mock()

        # Test
        processor = ServiceFactory.create_inference_processor("test-network")

        # Assertions
        mock_load_config.assert_called_once_with("test-network")
        mock_load_prompt.assert_called_once()
        mock_load_format.assert_called_once()
        assert processor is not None
        assert processor.network_id == "test-network"
        assert processor.system_prompt == "test prompt"
        assert processor.response_format == {"test": "format"}

    @patch("app.services.service_factory.ServiceFactory._load_network_config")
    @patch("app.services.multi_stream_processor.MultiStreamProcessor")
    def test_create_multi_stream_processor(self, mock_processor_class, mock_load_config):
        """Test creating a MultiStreamProcessor."""
        # Setup mocks
        mock_config = {"test": "config"}
        mock_load_config.return_value = mock_config
        mock_processor_instance = Mock()
        mock_processor_class.return_value = mock_processor_instance

        # Test
        processor = ServiceFactory.create_multi_stream_processor("test-network")

        # Assertions
        mock_load_config.assert_called_once_with("test-network")
        mock_processor_class.assert_called_once_with("test-network", mock_config)
        assert processor == mock_processor_instance

    @patch("app.services.service_factory.ServiceFactory._load_network_config")
    @patch("app.services.sentence_processor.SentenceProcessor")
    def test_create_sentence_processor(self, mock_processor_class, mock_load_config):
        """Test creating a SentenceProcessor."""
        # Setup mocks
        mock_config = {"test": "config"}
        mock_load_config.return_value = mock_config
        mock_processor_instance = Mock()
        mock_processor_class.return_value = mock_processor_instance

        # Test
        processor = ServiceFactory.create_sentence_processor("test-network")

        # Assertions
        mock_load_config.assert_called_once_with("test-network")
        mock_processor_class.assert_called_once_with("test-network", mock_config)
        assert processor == mock_processor_instance

    def test_load_network_config(self):
        """Test loading network configuration."""
        config = ServiceFactory._load_network_config("test-network")

        assert isinstance(config, dict)
        assert "model_name" in config
        assert "system_prompt" in config
        assert "wake_words" in config
        assert "sample_rate" in config
        assert "max_clients" in config

    @patch("builtins.open", side_effect=FileNotFoundError)
    def test_load_default_system_prompt_file_not_found(self, mock_open):
        """Test loading default system prompt when file is not found."""
        prompt = ServiceFactory._load_default_system_prompt()
        assert prompt == "You are a helpful AI assistant."

    @patch("builtins.open", mock_open_with_content("Test system prompt"))
    def test_load_default_system_prompt_success(self):
        """Test loading default system prompt successfully."""
        prompt = ServiceFactory._load_default_system_prompt()
        assert prompt == "Test system prompt"

    @patch("builtins.open", side_effect=FileNotFoundError)
    def test_load_action_system_prompt_file_not_found(self, mock_open):
        """Test loading action system prompt when file is not found."""
        prompt = ServiceFactory._load_action_system_prompt()
        assert prompt == "You are a helpful AI assistant for action processing."

    @patch("builtins.open", mock_open_with_content("Test action prompt"))
    def test_load_action_system_prompt_success(self):
        """Test loading action system prompt successfully."""
        prompt = ServiceFactory._load_action_system_prompt()
        assert prompt == "Test action prompt"

    @patch("builtins.open", side_effect=FileNotFoundError)
    def test_load_action_response_format_file_not_found(self, mock_open):
        """Test loading action response format when file is not found."""
        format_dict = ServiceFactory._load_action_response_format()
        assert format_dict == {}

    @patch("builtins.open", mock_open_with_content('{"test": "format"}'))
    @patch("json.load")
    def test_load_action_response_format_success(self, mock_json_load):
        """Test loading action response format successfully."""
        mock_json_load.return_value = {"test": "format"}
        format_dict = ServiceFactory._load_action_response_format()
        assert format_dict == {"test": "format"}

    # _load_model_tools method no longer exists - tools are now registered directly in processors


class TestServiceRegistry:
    """Test cases for ServiceRegistry class."""

    def test_get_service_new_service(self):
        """Test getting a new service that doesn't exist in registry."""
        # Clear registry
        service_registry._services.clear()
        service_registry._configs.clear()

        mock_factory = Mock()
        mock_service = Mock()
        mock_factory.return_value = mock_service

        result = service_registry.get_service("test-network", "test-service", mock_factory)

        assert result == mock_service
        assert service_registry._services["test-network"]["test-service"] == mock_service
        mock_factory.assert_called_once_with("test-network")

    def test_get_service_existing_service(self):
        """Test getting an existing service from registry."""
        # Clear registry
        service_registry._services.clear()
        service_registry._configs.clear()

        # Initialize network first
        service_registry._services["test-network"] = {}
        from datetime import datetime, timezone

        service_registry._configs["test-network"] = Mock()
        service_registry._configs["test-network"].last_accessed = datetime.now(timezone.utc)
        service_registry._configs["test-network"].ttl_seconds = 3600

        mock_service = Mock()
        service_registry._services["test-network"]["test-service"] = mock_service

        mock_factory = Mock()

        result = service_registry.get_service("test-network", "test-service", mock_factory)

        assert result == mock_service
        mock_factory.assert_not_called()

    def test_cleanup_all(self):
        """Test cleaning up all services from registry."""
        # Add some services
        service_registry._services["network1"] = {"service1": Mock()}
        service_registry._services["network2"] = {"service2": Mock()}
        service_registry._configs["network1"] = Mock()
        service_registry._configs["network2"] = Mock()

        service_registry.cleanup_all()

        assert len(service_registry._services) == 0


class TestServiceFactoryConvenienceFunctions:
    """Test cases for ServiceFactory convenience functions."""

    @patch("app.services.service_factory.service_registry.get_service")
    def test_get_command_processor(self, mock_get_service):
        """Test get_command_processor convenience function."""
        mock_service = Mock()
        mock_get_service.return_value = mock_service

        result = get_command_processor("test-network")

        mock_get_service.assert_called_once_with(
            "test-network", "command_processor", ServiceFactory.create_command_processor
        )
        assert result == mock_service

    @patch("app.services.service_factory.service_registry.get_service")
    def test_get_sentence_processor(self, mock_get_service):
        """Test get_sentence_processor convenience function."""
        mock_service = Mock()
        mock_get_service.return_value = mock_service

        result = get_sentence_processor("test-network")

        mock_get_service.assert_called_once_with(
            "test-network", "sentence_processor", ServiceFactory.create_sentence_processor
        )
        assert result == mock_service
