"""
Unit tests for configuration.
"""

import pytest
from unittest.mock import patch
from app.core.config import Settings, settings


class TestSettings:
    """Test cases for Settings class."""

    def test_default_values(self):
        """Test that default values are set correctly."""
        test_settings = Settings()

        assert test_settings.app_name == "Mira Backend"
        assert test_settings.app_version == "4.3.0"
        assert test_settings.debug is False
        assert test_settings.database_url == "sqlite:///./mira.db"
        assert test_settings.lm_studio_url == "http://localhost:1234/v1"
        assert test_settings.lm_studio_api_key == "lm-studio"
        assert test_settings.cors_origins == ["*"]
        assert test_settings.log_level == "INFO"
        assert test_settings.sample_rate == 16000
        assert test_settings.conversation_gap_threshold == 300
        assert test_settings.context_similarity_threshold == 0.7

    def test_environment_variable_override(self):
        """Test that environment variables can override default values."""
        with patch.dict('os.environ', {
            'APP_NAME': 'Test App',
            'DEBUG': 'true',
            'DATABASE_URL': 'postgresql://test:test@localhost/test',
            'LOG_LEVEL': 'DEBUG'
        }):
            test_settings = Settings()

            assert test_settings.app_name == "Test App"
            assert test_settings.debug is True
            assert test_settings.database_url == "postgresql://test:test@localhost/test"
            assert test_settings.log_level == "DEBUG"

    def test_cors_origins_parsing(self):
        """Test that CORS origins are properly parsed from environment."""
        with patch.dict('os.environ', {
            'CORS_ORIGINS': '["http://localhost:3000", "https://example.com", "https://app.example.com"]'
        }):
            test_settings = Settings()

            assert test_settings.cors_origins == [
                "http://localhost:3000",
                "https://example.com",
                "https://app.example.com"
            ]

    def test_numeric_values(self):
        """Test that numeric values are properly parsed."""
        with patch.dict('os.environ', {
            'SAMPLE_RATE': '22050',
            'CONVERSATION_GAP_THRESHOLD': '600',
            'CONTEXT_SIMILARITY_THRESHOLD': '0.8'
        }):
            test_settings = Settings()

            assert test_settings.sample_rate == 22050
            assert test_settings.conversation_gap_threshold == 600
            assert test_settings.context_similarity_threshold == 0.8

    def test_boolean_values(self):
        """Test that boolean values are properly parsed."""
        with patch.dict('os.environ', {
            'DEBUG': 'true',
            'SERVICE_ENABLED': 'false'
        }):
            test_settings = Settings()

            assert test_settings.debug is True

    def test_config_class_attributes(self):
        """Test that the Config class has the correct attributes."""
        config = Settings.Config()

        assert config.env_file == ".env"
        assert config.case_sensitive is False

    def test_global_settings_instance(self):
        """Test that the global settings instance is properly configured."""
        assert isinstance(settings, Settings)
        assert settings.app_name == "Mira Backend"
        assert settings.app_version == "4.3.0"

    def test_field_validation(self):
        """Test that field validation works correctly."""
        # Test with invalid values
        with patch.dict('os.environ', {
            'SAMPLE_RATE': 'invalid_number',
            'CONTEXT_SIMILARITY_THRESHOLD': 'not_a_float'
        }):
            with pytest.raises(ValueError):
                Settings()

    def test_optional_fields(self):
        """Test that optional fields work correctly."""
        # Test with minimal environment variables
        with patch.dict('os.environ', {}, clear=True):
            test_settings = Settings()

            # Should use defaults
            assert test_settings.app_name == "Mira Backend"
            assert test_settings.database_url == "sqlite:///./mira.db"

    def test_env_file_loading(self):
        """Test that .env file is loaded if it exists."""
        # This test would require creating a temporary .env file
        # For now, we'll test that the configuration is set to load it
        config = Settings.Config()
        assert config.env_file == ".env"

    def test_case_insensitive_environment_variables(self):
        """Test that environment variables are case insensitive."""
        with patch.dict('os.environ', {
            'app_name': 'Test App Case Insensitive',
            'DATABASE_URL': 'sqlite:///test.db'
        }):
            test_settings = Settings()

            assert test_settings.app_name == "Test App Case Insensitive"
            assert test_settings.database_url == "sqlite:///test.db"
