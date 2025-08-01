from fastapi.testclient import TestClient
from mira import app
import json

client = TestClient(app)


def test_root_status():
    """Test server status endpoint."""
    response = client.get("/")
    assert response.status_code == 200
    data = response.json()
    assert "enabled" in data
    assert "stream_scoring" in data["features"]
    assert data["features"]["stream_scoring"] is True


def test_register_client():
    """Test client registration endpoint."""
    response = client.post("/service/client/register/TestClient")
    assert response.status_code in (200, 201)
    data = response.json()
    assert "stream_scoring_enabled" in data
    assert data["stream_scoring_enabled"] is True


def test_register_client_with_metadata():
    """Test client registration with device type and location."""
    response = client.post(
        "/service/client/register/TestClientWithMetadata",
        params={"device_type": "phone"},
        json={"location": {"lat": 37.7749, "lng": -122.4194}}
    )
    assert response.status_code in (200, 201)
    data = response.json()
    assert "stream_scoring_enabled" in data


def test_deregister_client():
    """Test client deregistration endpoint."""
    # First register a client
    client.post("/service/client/register/TestClientDeregister")
    
    # Then deregister
    response = client.delete("/service/client/deregister/TestClientDeregister")
    assert response.status_code == 200
    data = response.json()
    assert "stream_scoring_removed" in data


def test_enable_disable_transcription():
    """Test enabling and disabling transcription endpoints."""
    enable = client.patch("service/enable")
    assert enable.status_code == 200
    disable = client.patch("service/disable")
    assert disable.status_code == 200


def test_process_interaction_empty():
    """Test processing interaction with empty body (should fail)."""
    response = client.post("/interactions/register", json={})
    assert response.status_code in (400, 422)


def test_get_recent_interaction():
    """Test getting recent interactions."""
    response = client.get("/interactions?limit=10")
    assert response.status_code == 200
    assert isinstance(response.json(), list)


# ============ Audio Stream Scoring Tests ============

def test_get_best_stream_no_clients():
    """Test getting best stream when no clients are active."""
    response = client.get("/streams/best")
    assert response.status_code == 200
    data = response.json()
    assert "best_stream" in data
    # May be None if no streams are available


def test_get_all_stream_scores_empty():
    """Test getting all stream scores when no streams are active."""
    response = client.get("/streams/scores")
    assert response.status_code == 200
    data = response.json()
    assert "active_streams" in data
    assert "stream_scores" in data
    assert "current_best" in data
    assert isinstance(data["stream_scores"], dict)


def test_set_phone_distance_nonexistent_client():
    """Test setting phone distance for non-existent client."""
    response = client.post(
        "/streams/nonexistent_client/distance",
        json={"distance": 5.0}
    )
    assert response.status_code == 404


def test_get_client_stream_info_nonexistent():
    """Test getting stream info for non-existent client."""
    response = client.get("/streams/nonexistent_client/info")
    assert response.status_code == 404


def test_cleanup_inactive_streams():
    """Test cleanup of inactive streams."""
    response = client.post("/streams/cleanup?timeout_seconds=1")
    assert response.status_code == 200
    data = response.json()
    assert "removed_clients" in data
    assert "timeout_seconds" in data
    assert isinstance(data["removed_clients"], list)


def test_full_stream_scoring_workflow():
    """Test complete workflow with stream scoring."""
    # 1. Register a client with metadata
    register_response = client.post(
        "/service/client/register/WorkflowTestClient",
        params={"device_type": "phone"},
        json={"location": {"lat": 37.7749, "lng": -122.4194}}
    )
    assert register_response.status_code in (200, 201)
    
    # 2. Set phone distance
    distance_response = client.post(
        "/streams/WorkflowTestClient/distance",
        json={"distance": 3.5}
    )
    assert distance_response.status_code == 200
    
    # 3. Get client info
    info_response = client.get("/streams/WorkflowTestClient/info")
    assert info_response.status_code == 200
    info_data = info_response.json()
    assert info_data["client_id"] == "WorkflowTestClient"
    assert info_data["device_type"] == "phone"
    assert info_data["quality_metrics"]["phone_distance"] == 3.5
    
    # 4. Check stream scores
    scores_response = client.get("/streams/scores")
    assert scores_response.status_code == 200
    scores_data = scores_response.json()
    assert "WorkflowTestClient" in scores_data["stream_scores"]
    
    # 5. Check best stream
    best_response = client.get("/streams/best")
    assert best_response.status_code == 200
    
    # 6. Deregister client
    deregister_response = client.delete("/service/client/deregister/WorkflowTestClient")
    assert deregister_response.status_code == 200
    
    # 7. Verify client is gone
    final_info_response = client.get("/streams/WorkflowTestClient/info")
    assert final_info_response.status_code == 404
