"""
Task management tests.
"""
import pytest
from fastapi.testclient import TestClient
from sqlalchemy.orm import Session

from app.main import app
from app.models.user import User
from app.models.action import Action
from app.core.security import get_password_hash

client = TestClient(app)


@pytest.fixture
def test_user_with_token(db: Session):
    """Create a test user and return token."""
    user = User(
        username="taskuser",
        email="task@example.com",
        hashed_password=get_password_hash("taskpassword"),
        is_active=True
    )
    db.add(user)
    db.commit()
    db.refresh(user)

    # Login to get token
    login_response = client.post(
        "/api/v1/auth/login",
        data={
            "username": "taskuser",
            "password": "taskpassword"
        }
    )
    token = login_response.json()["access_token"]

    return user, token


def test_create_action(test_user_with_token):
    """Test creating an action."""
    user, token = test_user_with_token

    response = client.post(
        "/api/v1/tasks/",
        json={
            "user_id": str(user.id),
            "action_type": "reminder",
            "details": "Test reminder"
        },
        headers={"Authorization": f"Bearer {token}"}
    )
    assert response.status_code == 200
    data = response.json()
    assert data["action_type"] == "reminder"
    assert data["details"] == "Test reminder"
    assert data["status"] == "pending"


def test_get_actions(test_user_with_token):
    """Test getting user actions."""
    user, token = test_user_with_token

    response = client.get(
        "/api/v1/tasks/",
        headers={"Authorization": f"Bearer {token}"}
    )
    assert response.status_code == 200
    data = response.json()
    assert isinstance(data, list)


def test_get_action(test_user_with_token):
    """Test getting a specific action."""
    user, token = test_user_with_token

    # First create an action
    create_response = client.post(
        "/api/v1/tasks/",
        json={
            "user_id": str(user.id),
            "action_type": "reminder",
            "details": "Test reminder"
        },
        headers={"Authorization": f"Bearer {token}"}
    )
    action_id = create_response.json()["id"]

    # Then get it
    response = client.get(
        f"/api/v1/tasks/{action_id}",
        headers={"Authorization": f"Bearer {token}"}
    )
    assert response.status_code == 200
    data = response.json()
    assert data["id"] == action_id


def test_update_action(test_user_with_token):
    """Test updating an action."""
    user, token = test_user_with_token

    # First create an action
    create_response = client.post(
        "/api/v1/tasks/",
        json={
            "user_id": str(user.id),
            "action_type": "reminder",
            "details": "Test reminder"
        },
        headers={"Authorization": f"Bearer {token}"}
    )
    action_id = create_response.json()["id"]

    # Then update it
    response = client.put(
        f"/api/v1/tasks/{action_id}",
        json={
            "status": "completed",
            "details": "Updated reminder"
        },
        headers={"Authorization": f"Bearer {token}"}
    )
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "completed"
    assert data["details"] == "Updated reminder"


def test_delete_action(test_user_with_token):
    """Test deleting an action."""
    user, token = test_user_with_token

    # First create an action
    create_response = client.post(
        "/api/v1/tasks/",
        json={
            "user_id": str(user.id),
            "action_type": "reminder",
            "details": "Test reminder"
        },
        headers={"Authorization": f"Bearer {token}"}
    )
    action_id = create_response.json()["id"]

    # Then delete it
    response = client.delete(
        f"/api/v1/tasks/{action_id}",
        headers={"Authorization": f"Bearer {token}"}
    )
    assert response.status_code == 200
    data = response.json()
    assert data["detail"] == "Action deleted successfully"


def test_unauthorized_access():
    """Test accessing tasks without authentication."""
    response = client.get("/api/v1/tasks/")
    assert response.status_code == 401

