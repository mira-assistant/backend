"""
Tests for assistant endpoints.
"""

import pytest
from fastapi import status
import io


def test_register_interaction(client, test_network):
    """Test registering a new interaction."""
    # Create dummy audio file
    audio_data = io.BytesIO(b"dummy audio data")
    audio_data.name = "test.wav"

    response = client.post(
        f"/api/v1/assistant/networks/{test_network.network_id}/interactions/register",
        params={"password": "testpassword123"},
        files={"audio": ("test.wav", audio_data, "audio/wav")}
    )
    assert response.status_code == status.HTTP_200_OK
    data = response.json()
    assert "id" in data


def test_register_interaction_wrong_password(client, test_network):
    """Test registering interaction with wrong password."""
    audio_data = io.BytesIO(b"dummy audio data")
    audio_data.name = "test.wav"

    response = client.post(
        f"/api/v1/assistant/networks/{test_network.network_id}/interactions/register",
        params={"password": "wrongpassword"},
        files={"audio": ("test.wav", audio_data, "audio/wav")}
    )
    assert response.status_code == status.HTTP_401_UNAUTHORIZED


def test_register_interaction_disabled_service(client, test_network, test_db):
    """Test registering interaction when service is disabled."""
    # Disable service
    test_network.service_enabled = False
    test_db.commit()

    audio_data = io.BytesIO(b"dummy audio data")
    audio_data.name = "test.wav"

    response = client.post(
        f"/api/v1/assistant/networks/{test_network.network_id}/interactions/register",
        params={"password": "testpassword123"},
        files={"audio": ("test.wav", audio_data, "audio/wav")}
    )
    assert response.status_code == status.HTTP_400_BAD_REQUEST


def test_list_interactions(client, test_network):
    """Test listing interactions."""
    response = client.get(
        f"/api/v1/assistant/networks/{test_network.network_id}/interactions",
        params={"password": "testpassword123"}
    )
    assert response.status_code == status.HTTP_200_OK
    data = response.json()
    assert isinstance(data, list)