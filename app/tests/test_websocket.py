"""
Tests for WebSocket functionality.
"""

import pytest
from fastapi.testclient import TestClient
from fastapi import status
import json


def test_websocket_connection(client: TestClient, test_network, test_client):
    """Test WebSocket connection."""
    with client.websocket_connect(
        f"/ws/{test_network.network_id}/{test_client.client_id}",
        params={"auth_token": test_client.auth_token}
    ) as websocket:
        # Should receive ping
        data = websocket.receive_json()
        assert data["type"] == "ping"


def test_websocket_unauthorized(client: TestClient, test_network, test_client):
    """Test WebSocket connection with wrong token."""
    with pytest.raises(Exception):
        with client.websocket_connect(
            f"/ws/{test_network.network_id}/{test_client.client_id}",
            params={"auth_token": "wrongtoken"}
        ):
            pass


def test_websocket_notifications(client: TestClient, test_network, test_client):
    """Test WebSocket notifications for interactions."""
    # Connect to WebSocket
    with client.websocket_connect(
        f"/ws/{test_network.network_id}/{test_client.client_id}",
        params={"auth_token": test_client.auth_token}
    ) as websocket:
        # Skip initial ping
        websocket.receive_json()

        # Register interaction
        audio_data = io.BytesIO(b"dummy audio data")
        audio_data.name = "test.wav"

        response = client.post(
            f"/api/v1/assistant/networks/{test_network.network_id}/interactions/register",
            params={"auth_token": test_client.auth_token},
            files={"audio": ("test.wav", audio_data, "audio/wav")},
            data={"client_id": test_client.client_id}
        )
        assert response.status_code == status.HTTP_200_OK

        # Should receive interaction notification
        data = websocket.receive_json()
        assert data["type"] == "new_interaction"
        assert "interaction" in data
        assert "id" in data["interaction"]
