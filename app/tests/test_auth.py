"""
Authentication tests.
"""
import pytest
from fastapi.testclient import TestClient
from sqlalchemy.orm import Session

from app.main import app
from app.db.session import get_db
from app.models.user import User
from app.core.security import get_password_hash


client = TestClient(app)


@pytest.fixture
def test_user(db: Session):
    """Create a test user."""
    user = User(
        username="testuser",
        email="test@example.com",
        hashed_password=get_password_hash("testpassword"),
        is_active=True
    )
    db.add(user)
    db.commit()
    db.refresh(user)
    return user


def test_register_user(db):
    """Test user registration."""
    response = client.post(
        "/api/v1/auth/register",
        json={
            "username": "newuser",
            "email": "newuser@example.com",
            "password": "newpassword"
        }
    )
    assert response.status_code == 200
    data = response.json()
    assert data["username"] == "newuser"
    assert data["email"] == "newuser@example.com"
    assert "id" in data


def test_register_duplicate_user(db):
    """Test registration with duplicate username."""
    # First registration
    client.post(
        "/api/v1/auth/register",
        json={
            "username": "duplicate",
            "email": "duplicate1@example.com",
            "password": "password"
        }
    )

    # Second registration with same username
    response = client.post(
        "/api/v1/auth/register",
        json={
            "username": "duplicate",
            "email": "duplicate2@example.com",
            "password": "password"
        }
    )
    assert response.status_code == 400


def test_login_user(test_user):
    """Test user login."""
    response = client.post(
        "/api/v1/auth/login",
        data={
            "username": "testuser",
            "password": "testpassword"
        }
    )
    assert response.status_code == 200
    data = response.json()
    assert "access_token" in data
    assert data["token_type"] == "bearer"


def test_login_invalid_credentials(db):
    """Test login with invalid credentials."""
    response = client.post(
        "/api/v1/auth/login",
        data={
            "username": "nonexistent",
            "password": "wrongpassword"
        }
    )
    assert response.status_code == 401


def test_get_current_user(test_user):
    """Test getting current user info."""
    # First login to get token
    login_response = client.post(
        "/api/v1/auth/login",
        data={
            "username": "testuser",
            "password": "testpassword"
        }
    )
    token = login_response.json()["access_token"]

    # Use token to get user info
    response = client.get(
        "/api/v1/auth/me",
        headers={"Authorization": f"Bearer {token}"}
    )
    assert response.status_code == 200
    data = response.json()
    assert data["username"] == "testuser"

