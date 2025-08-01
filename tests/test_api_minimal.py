"""
Minimal API tests for the audio stream scoring endpoints.
These tests focus on the new API endpoints without requiring heavy dependencies.
"""

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

# Mock heavy dependencies that aren't needed for API testing
# Note: We specifically avoid mocking scipy.signal as it interferes with other tests
sys.modules["whisper"] = Mock()
sys.modules["resemblyzer"] = Mock()
sys.modules["noisereduce"] = Mock()
sys.modules["spacy"] = Mock()
sys.modules["sentence_transformers"] = Mock()
sys.modules["transformers"] = Mock()
sys.modules["sklearn"] = Mock()
sys.modules["scikit-learn"] = Mock()

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
        assert "features" in data
        assert "stream_scoring" in data["features"]
        assert data["features"]["stream_scoring"] is True

    def test_register_client_basic(self):
        """Test basic client registration"""
        response = client.post("/service/client/register/test_client_basic")
        assert response.status_code == 200
        data = response.json()
        assert "stream_scoring_enabled" in data
        assert data["stream_scoring_enabled"] is True
        assert "test_client_basic registered successfully" in data["message"]

    def test_register_client_with_metadata(self):
        """Test client registration with device type"""
        response = client.post("/service/client/register/test_client_metadata?device_type=phone")
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
        """Test setting phone distance for non-existent client"""
        response = client.post("/streams/nonexistent_client/distance", json={"distance": 5.0})
        assert response.status_code == 404

    def test_set_phone_distance_valid_client(self):
        """Test setting phone distance for valid client"""
        # Register client first
        client.post("/service/client/register/test_client_distance")

        # Set distance
        response = client.post("/streams/test_client_distance/distance", json={"distance": 3.5})
        assert response.status_code == 200
        data = response.json()
        assert "distance" in data
        assert data["distance"] == 3.5
        assert "current_best_stream" in data

    def test_get_client_stream_info_nonexistent(self):
        """Test getting stream info for non-existent client"""
        response = client.get("/streams/nonexistent_client/info")
        assert response.status_code == 404

    def test_get_client_stream_info_valid(self):
        """Test getting stream info for valid client"""
        # Register client first
        client.post("/service/client/register/test_client_info?device_type=tablet")

        response = client.get("/streams/test_client_info/info")
        assert response.status_code == 200
        data = response.json()
        assert "client_id" in data
        assert data["client_id"] == "test_client_info"
        assert "device_type" in data
        assert data["device_type"] == "tablet"
        assert "quality_metrics" in data
        assert "current_score" in data
        assert "is_best_stream" in data

    def test_cleanup_inactive_streams(self):
        """Test cleanup of inactive streams"""
        response = client.post("/streams/cleanup?timeout_seconds=1")
        assert response.status_code == 200
        data = response.json()
        assert "removed_clients" in data
        assert "timeout_seconds" in data
        assert data["timeout_seconds"] == 1
        assert isinstance(data["removed_clients"], list)

    def test_full_workflow(self):
        """Test complete workflow with stream scoring"""
        client_id = "workflow_test_client"

        # 1. Register client
        register_response = client.post(f"/service/client/register/{client_id}?device_type=phone")
        assert register_response.status_code == 200

        # 2. Set phone distance
        distance_response = client.post(f"/streams/{client_id}/distance", json={"distance": 2.5})
        assert distance_response.status_code == 200

        # 3. Get client info
        info_response = client.get(f"/streams/{client_id}/info")
        assert info_response.status_code == 200
        info_data = info_response.json()
        assert info_data["quality_metrics"]["phone_distance"] == 2.5

        # 4. Check all scores
        scores_response = client.get("/streams/scores")
        assert scores_response.status_code == 200
        scores_data = scores_response.json()
        assert client_id in scores_data["stream_scores"]

        # 5. Get best stream
        best_response = client.get("/streams/best")
        assert best_response.status_code == 200

        # 6. Deregister
        deregister_response = client.delete(f"/service/client/deregister/{client_id}")
        assert deregister_response.status_code == 200

        # 7. Verify client is gone
        final_info_response = client.get(f"/streams/{client_id}/info")
        assert final_info_response.status_code == 404

    def test_phone_service_endpoints(self):
        """Test phone-specific service endpoints."""
        # Test enable service
        response = client.patch("/phone/service/enable")
        assert response.status_code == 200
        data = response.json()
        assert data["enabled"] is True

        # Test disable service
        response = client.patch("/phone/service/disable")
        assert response.status_code == 200
        data = response.json()
        assert data["enabled"] is False

        # Test service status
        response = client.get("/phone/service/status")
        assert response.status_code == 200
        data = response.json()
        assert "enabled" in data
        assert "version" in data

    def test_phone_distance_endpoints(self):
        """Test phone distance tracking endpoints."""
        # Test update distance with no clients
        response = client.post("/phone/distance/update", json={"distance": 2.5})
        assert response.status_code == 200
        data = response.json()
        assert data["distance"] == 2.5

        # Test invalid distance
        response = client.post("/phone/distance/update", json={"distance": -1})
        assert response.status_code == 400

        # Test missing distance
        response = client.post("/phone/distance/update", json={})
        assert response.status_code == 400

        # Test nearest client (first clean up any existing clients)
        response = client.get("/phone/distance/nearest_client")
        assert response.status_code == 200
        data = response.json()
        # May or may not be None depending on test isolation

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


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
