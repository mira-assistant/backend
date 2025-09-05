"""
Tests for authentication endpoints.
"""

import pytest
from fastapi import status


def test_register_network(client):
    """Test network registration."""
    response = client.post(
        "/api/v1/auth/register",
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


def test_login_network(client, test_network):
    """Test network login."""
    response = client.post(
        "/api/v1/auth/login",
        json={
            "network_id": test_network.network_id,
            "password": "testpassword123"
        }
    )
    assert response.status_code == status.HTTP_200_OK
    data = response.json()
    assert data["network_id"] == test_network.network_id


def test_login_wrong_password(client, test_network):
    """Test login with wrong password."""
    response = client.post(
        "/api/v1/auth/login",
        json={
            "network_id": test_network.network_id,
            "password": "wrongpassword"
        }
    )
    assert response.status_code == status.HTTP_401_UNAUTHORIZED


def test_login_nonexistent_network(client):
    """Test login with nonexistent network."""
    response = client.post(
        "/api/v1/auth/login",
        json={
            "network_id": "nonexistent",
            "password": "testpassword123"
        }
    )
    assert response.status_code == status.HTTP_404_NOT_FOUND