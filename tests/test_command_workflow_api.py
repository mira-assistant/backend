"""
Integration tests for command workflow API endpoints
"""

import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import pytest
from unittest.mock import patch, Mock, MagicMock
import json

# Mock heavy dependencies before importing mira
sys.modules['context_processor'] = MagicMock()
sys.modules['sentence_processor'] = MagicMock()

# Mock the processor creation
mock_processor = MagicMock()
with patch('context_processor.create_context_processor', return_value=mock_processor):
    from mira import app
    from fastapi.testclient import TestClient


class TestCommandWorkflowAPI:
    def setup_method(self):
        self.client = TestClient(app)

    def test_get_last_command_result_empty(self):
        """Test getting last command result when none exists"""
        response = self.client.get("/commands/last-result")
        assert response.status_code == 200
        data = response.json()
        assert "last_command_result" in data

    def test_get_available_callbacks(self):
        """Test getting available callback functions"""
        response = self.client.get("/commands/callbacks")
        assert response.status_code == 200
        data = response.json()
        
        assert "available_functions" in data
        assert "function_descriptions" in data
        
        # Check for default callbacks
        functions = data["available_functions"]
        assert "getWeather" in functions
        assert "getTime" in functions
        assert "disableMira" in functions
        
        descriptions = data["function_descriptions"]
        assert isinstance(descriptions, dict)
        assert len(descriptions) >= 3


if __name__ == "__main__":
    pytest.main([__file__, "-v"])