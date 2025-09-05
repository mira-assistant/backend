"""
Tests for network management endpoints.
"""

import pytest
from fastapi import status


def test_create_network(client):
    """Test network creation."""
    response = client.post(
        "/api/v1/networks/",
        json={
            "name": "Test Network",
            "password": "testpassword123"
        }
    )
    assert response.status_code == status.HTTP_200_OK
    data = response.json()
    assert data["name"] == "Test Network"
    assert "network_id" in data
    assert data["is_active"]


def test_authenticate_network(client, test_network):
    """Test network authentication."""
    response = client.post(
        "/api/v1/networks/auth",
        json={
            "network_id": test_network.network_id,
            "password": "testpassword123"
        }
    )
    assert response.status_code == status.HTTP_200_OK
    data = response.json()
    assert data["network_id"] == test_network.network_id


def test_authenticate_network_wrong_password(client, test_network):
    """Test network authentication with wrong password."""
    response = client.post(
        "/api/v1/networks/auth",
        json={
            "network_id": test_network.network_id,
            "password": "wrongpassword"
        }
    )
    assert response.status_code == status.HTTP_401_UNAUTHORIZED


def test_get_network(client, test_network, test_client):
    """Test getting network details."""
    response = client.get(
        f"/api/v1/networks/{test_network.network_id}",
        params={"auth_token": test_client.auth_token}
    )
    assert response.status_code == status.HTTP_200_OK
    data = response.json()
    assert data["network_id"] == test_network.network_id
    assert data["name"] == test_network.name


def test_get_network_wrong_token(client, test_network):
    """Test getting network details with wrong token."""
    response = client.get(
        f"/api/v1/networks/{test_network.network_id}",
        params={"auth_token": "wrongtoken"}
    )
    assert response.status_code == status.HTTP_401_UNAUTHORIZED


def test_update_network(client, test_network, test_client):
    """Test updating network settings."""
    response = client.put(
        f"/api/v1/networks/{test_network.network_id}",
        params={"auth_token": test_client.auth_token},
        json={
            "name": "Updated Network",
            "service_enabled": True
        }
    )
    assert response.status_code == status.HTTP_200_OK
    data = response.json()
    assert data["name"] == "Updated Network"
    assert data["service_enabled"]


def test_update_password(client, test_network, test_client):
    """Test updating network password."""
    response = client.post(
        f"/api/v1/networks/{test_network.network_id}/password",
        params={"auth_token": test_client.auth_token},
        json={
            "current_password": "testpassword123",
            "new_password": "newpassword123"
        }
    )
    assert response.status_code == status.HTTP_200_OK

    # Try authenticating with new password
    response = client.post(
        "/api/v1/networks/auth",
        json={
            "network_id": test_network.network_id,
            "password": "newpassword123"
        }
    )
    assert response.status_code == status.HTTP_200_OK


def test_delete_network(client, test_network, test_client):
    """Test deleting a network."""
    response = client.delete(
        f"/api/v1/networks/{test_network.network_id}",
        params={"auth_token": test_client.auth_token}
    )
    assert response.status_code == status.HTTP_200_OK

    # Try getting deleted network
    response = client.get(
        f"/api/v1/networks/{test_network.network_id}",
        params={"auth_token": test_client.auth_token}
    )
    assert response.status_code == status.HTTP_404_NOT_FOUND
