from fastapi.testclient import TestClient
from mira import app

client = TestClient(app)


def test_root_status():
    """Test server status endpoint."""
    response = client.get("/")
    assert response.status_code == 200
    assert "enabled" in response.json()


def test_register_client():
    """Test client registration endpoint."""
    response = client.post("/register_client", params={"client_id": "TestClient"})
    assert response.status_code in (200, 201, 409)  # 409 if already registered
    data = response.json()
    assert "enabled" in data


def test_enable_disable_transcription():
    """Test enabling and disabling transcription endpoints."""
    enable = client.patch("/enable")
    assert enable.status_code == 200
    disable = client.patch("/disable")
    assert disable.status_code == 200


def test_process_interaction_empty():
    """Test processing interaction with empty body (should fail)."""
    response = client.post("/register_interaction", json={})
    assert response.status_code in (400, 422)


def test_get_recent_interactions():
    """Test getting recent interactions."""
    response = client.get("/interactions?limit=10")
    assert response.status_code == 200
    assert isinstance(response.json(), list)


def test_get_interaction_history():
    """Test interaction history endpoint."""
    response = client.get("/context/history")
    assert response.status_code == 200
    assert isinstance(response.json(), list)


# def test_clear_all_interactions():
#     """Test clearing interactions."""
#     response = client.delete("/interactions")
#     assert response.status_code == 200
#     assert "deleted_count" in response.json()

# def test_get_speaker_summary():
#     """Test speaker summary endpoint."""
#     response = client.get("/context/speakers")
#     assert response.status_code == 200
#     assert isinstance(response.json(), list)


# @pytest.mark.parametrize("speaker_index,name", [(0, "Test User")])
# def test_identify_speaker(speaker_index, name):
#     """Test manual speaker identification."""
#     response = client.post("/context/identify_speaker", params={"speaker_index": speaker_index, "name": name})
#     assert response.status_code in (200, 404)
