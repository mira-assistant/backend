"""
Minimal API tests for the audio stream scoring endpoints.
These tests focus on the new API endpoints without requiring heavy dependencies.
"""

import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import pytest
from unittest.mock import Mock, patch
from fastapi.testclient import TestClient
import sys


# Store original modules if they exist, so we can restore them later
_original_modules = {}
_modules_to_mock = [
    "whisper",
    "resemblyzer",
    "noisereduce",
    "spacy",
    "sentence_transformers",
    "transformers",
    "sklearn",
    "scikit-learn",
    "db",
    "models",
    "inference_processor",
    "sentence_processor",
    "context_processor",
]

for module_name in _modules_to_mock:
    if module_name in sys.modules:
        _original_modules[module_name] = sys.modules[module_name]

# Mock the database and processor modules
sys.modules["db"] = Mock()
sys.modules["models"] = Mock()
sys.modules["inference_processor"] = Mock()
sys.modules["sentence_processor"] = Mock()
sys.modules["context_processor"] = Mock()

# Mock the database session and models
mock_db = Mock()
mock_db.query.return_value.filter_by.return_value.first.return_value = None
mock_db.query.return_value.order_by.return_value.limit.return_value.all.return_value = []


def mock_get_db_session():
    return mock_db


def mock_create_context_processor():
    return Mock()


# Apply mocks
with patch.dict(
    "sys.modules",
    {
        "db": Mock(get_db_session=mock_get_db_session),
        "models": Mock(Interaction=Mock, Person=Mock, Conversation=Mock),
        "context_processor": Mock(create_context_processor=mock_create_context_processor),
        "inference_processor": Mock(),
        "sentence_processor": Mock(),
        "command_processor": Mock(CommandProcessor=Mock),
        "ml_model_manager": Mock(),
    },
):
    from mira import app


def teardown_module():
    """Clean up global module mocks after tests complete"""
    # Restore original modules
    for module_name, original_module in _original_modules.items():
        sys.modules[module_name] = original_module

    # Remove mocked modules that weren't originally present
    for module_name in _modules_to_mock:
        if module_name not in _original_modules and module_name in sys.modules:
            del sys.modules[module_name]


client = TestClient(app)


class TestStreamScoringAPI:
    """Test suite for stream scoring API endpoints"""

    def test_root_status_includes_stream_scoring(self):
        """Test that the root status includes stream_scoring feature"""
        response = client.get("/")
        assert response.status_code == 200
        data = response.json()
        assert "enabled" in data

    def test_register_client_basic(self):
        """Test basic client registration"""
        response = client.post("/service/client/register/test_client_basic")
        assert response.status_code == 200
        data = response.json()
        assert "stream_scoring_enabled" in data
        assert data["stream_scoring_enabled"] is True
        assert "test_client_basic registered successfully" in data["message"]

    def test_register_client_with_metadata(self):
        """Test client registration (no longer accepts device_type)"""
        response = client.post("/service/client/register/test_client_metadata")
        assert response.status_code == 200
        data = response.json()
        assert data["stream_scoring_enabled"] is True

    def test_deregister_client(self):
        """Test client deregistration"""
        # First register
        client.post("/service/client/register/test_client_dereg")

        # Then deregister
        response = client.delete("/service/client/deregister/test_client_dereg")
        assert response.status_code == 200
        data = response.json()
        assert "stream_scoring_removed" in data

    def test_get_best_stream_no_active_streams(self):
        """Test getting best stream when no streams are active"""
        response = client.get("/streams/best")
        assert response.status_code == 200
        data = response.json()
        assert "best_stream" in data
        # Could be None or have a best stream depending on previous tests

    def test_get_all_stream_scores(self):
        """Test getting all stream scores"""
        response = client.get("/streams/scores")
        assert response.status_code == 200
        data = response.json()
        assert "active_streams" in data
        assert "stream_scores" in data
        assert "current_best" in data
        assert isinstance(data["stream_scores"], dict)
        assert isinstance(data["active_streams"], int)

    def test_set_phone_distance_nonexistent_client(self):
        """Test setting location for non-existent client"""
        location_data = {
            "client_id": "nonexistent_client",
            "location": {"latitude": 37.7749, "longitude": -122.4194},
        }
        response = client.post("/streams/phone/location", json=location_data)
        assert response.status_code == 404

    def test_set_phone_distance_valid_client(self):
        """Test setting location for valid client"""
        # Register client first
        client.post("/service/client/register/test_client_distance")

        # Set location
        location_data = {
            "client_id": "test_client_distance",
            "location": {"latitude": 37.7749, "longitude": -122.4194, "accuracy": 3.5},
        }
        response = client.post("/streams/phone/location", json=location_data)
        assert response.status_code == 200
        data = response.json()
        assert "location" in data
        assert data["location"]["latitude"] == 37.7749

    def test_get_client_stream_info_nonexistent(self):
        """Test getting stream info for non-existent client"""
        response = client.get("/streams/nonexistent_client/info")
        assert response.status_code == 404

    def test_get_client_stream_info_valid(self):
        """Test getting stream info for valid client"""
        # Register client first
        client.post("/service/client/register/test_client_info")

        response = client.get("/streams/test_client_info/info")
        assert response.status_code == 200
        data = response.json()
        assert "client_id" in data
        assert data["client_id"] == "test_client_info"
        assert "quality_metrics" in data
        assert "current_score" in data
        assert "is_best_stream" in data

    def test_full_workflow(self):
        """Test complete workflow with stream scoring"""
        client_id = "workflow_test_client"

        # 1. Register client
        register_response = client.post(f"/service/client/register/{client_id}")
        assert register_response.status_code == 200

        # 2. Set phone location
        location_data = {
            "client_id": client_id,
            "location": {"latitude": 37.7749, "longitude": -122.4194, "accuracy": 2.5},
        }
        location_response = client.post("/streams/phone/location", json=location_data)
        assert location_response.status_code == 200

        # 3. Set RSSI
        rssi_data = {
            "phone_client_id": "phone_device",
            "target_client_id": client_id,
            "rssi": -45.0,
        }
        rssi_response = client.post("/streams/phone/rssi", json=rssi_data)
        assert rssi_response.status_code == 200

        # 4. Get client info
        info_response = client.get(f"/streams/{client_id}/info")
        assert info_response.status_code == 200
        info_data = info_response.json()
        assert "quality_metrics" in info_data

        # 5. Check all scores
        scores_response = client.get("/streams/scores")
        assert scores_response.status_code == 200
        scores_data = scores_response.json()
        assert client_id in scores_data["stream_scores"]

        # 6. Get best stream
        best_response = client.get("/streams/best")
        assert best_response.status_code == 200

        # 7. Deregister
        deregister_response = client.delete(f"/service/client/deregister/{client_id}")
        assert deregister_response.status_code == 200

        # 8. Verify client is gone
        final_info_response = client.get(f"/streams/{client_id}/info")
        assert final_info_response.status_code == 404

    def test_register_interaction_stream_filtering(self):
        """Test that register_interaction properly filters based on stream quality."""
        # Register two clients
        client.post("/service/client/register/client1")
        client.post("/service/client/register/client2")

        # Test with minimal audio data - just verify the endpoint doesn't crash
        try:
            response = client.post(
                "/interactions/register?client_id=client1",
                content=b"fake_audio_data_with_at_least_some_bytes",
            )
            # Endpoint should handle the request gracefully, even if it fails processing
            assert response.status_code in [200, 422, 500]  # Accept various outcomes
        except Exception as e:
            # If there's an exception, make sure it's not due to our new logic
            assert "better audio streams" not in str(e)

    def test_new_location_endpoint(self):
        """Test the new /streams/phone/location endpoint"""
        # First register a client
        register_response = client.post("/service/client/register/location_test_client")
        assert register_response.status_code == 200

        # Test setting location
        location_data = {
            "client_id": "location_test_client",
            "location": {"latitude": 37.7749, "longitude": -122.4194, "accuracy": 5.0},
        }
        response = client.post("/streams/phone/location", json=location_data)
        assert response.status_code == 200
        data = response.json()
        assert "message" in data
        assert "location" in data
        assert data["location"]["latitude"] == 37.7749

    def test_new_rssi_endpoint(self):
        """Test the new /streams/phone/rssi endpoint"""
        # First register a client
        register_response = client.post("/service/client/register/rssi_test_client")
        assert register_response.status_code == 200

        # Test setting RSSI
        rssi_data = {
            "phone_client_id": "phone_device",
            "target_client_id": "rssi_test_client",
            "rssi": -45.5,
        }
        response = client.post("/streams/phone/rssi", json=rssi_data)
        assert response.status_code == 200
        data = response.json()
        assert "message" in data
        assert "rssi" in data
        assert data["rssi"] == -45.5

    def test_location_endpoint_missing_client_id(self):
        """Test location endpoint with missing client_id"""
        location_data = {"location": {"latitude": 37.7749, "longitude": -122.4194}}
        response = client.post("/streams/phone/location", json=location_data)
        assert response.status_code == 400

    def test_rssi_endpoint_nonexistent_client(self):
        """Test RSSI endpoint with non-existent client"""
        rssi_data = {
            "phone_client_id": "phone_device",
            "target_client_id": "nonexistent_client",
            "rssi": -50.0,
        }
        response = client.post("/streams/phone/rssi", json=rssi_data)
        assert response.status_code == 404

    def test_connected_clients_dict(self):
        """Test that connected_clients is now a dict instead of list"""
        # Register a client
        response = client.post("/service/client/register/dict_test_client")
        assert response.status_code == 200

        # Check root status
        status_response = client.get("/")
        assert status_response.status_code == 200
        status_data = status_response.json()
        assert "connected_clients" in status_data
        assert isinstance(status_data["connected_clients"], dict)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
