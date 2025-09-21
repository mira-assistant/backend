"""Integration tests for authentication endpoints."""

import json
import os
from datetime import datetime, timezone
from uuid import uuid4

import pytest
from fastapi import status
from fastapi.testclient import TestClient

import models
from core.security import create_access_token, get_password_hash


class TestAuthEndpoints:
    """Test authentication API endpoints."""

    def setup_method(self):
        """Set up test environment."""
        os.environ["JWT_SECRET_KEY"] = "test-secret-key-32-characters-minimum-length"

    def test_register_success(self, client: TestClient, db_session):
        """Test successful user registration."""
        user_data = {
            "email": "test@example.com",
            "password": "password123",
            "username": "testuser",
            "full_name": "Test User"
        }
        
        response = client.post("/api/v1/auth/register", json=user_data)
        
        assert response.status_code == status.HTTP_200_OK
        data = response.json()
        
        assert data["email"] == "test@example.com"
        assert data["username"] == "testuser"
        assert data["full_name"] == "Test User"
        assert data["is_active"] is True
        assert data["is_verified"] is False
        assert "id" in data
        assert "created_at" in data
        assert "hashed_password" not in data  # Should not expose password

    def test_register_duplicate_email(self, client: TestClient, db_session):
        """Test registration with duplicate email."""
        # Create existing user
        existing_user = models.User(
            email="test@example.com",
            hashed_password="hash"
        )
        db_session.add(existing_user)
        db_session.commit()
        
        user_data = {
            "email": "test@example.com",
            "password": "password123"
        }
        
        response = client.post("/api/v1/auth/register", json=user_data)
        
        assert response.status_code == status.HTTP_400_BAD_REQUEST
        assert "Email already registered" in response.json()["detail"]

    def test_register_invalid_email(self, client: TestClient):
        """Test registration with invalid email."""
        user_data = {
            "email": "invalid-email",
            "password": "password123"
        }
        
        response = client.post("/api/v1/auth/register", json=user_data)
        
        assert response.status_code == status.HTTP_422_UNPROCESSABLE_ENTITY

    def test_register_short_password(self, client: TestClient):
        """Test registration with short password."""
        user_data = {
            "email": "test@example.com",
            "password": "short"
        }
        
        response = client.post("/api/v1/auth/register", json=user_data)
        
        assert response.status_code == status.HTTP_422_UNPROCESSABLE_ENTITY

    def test_login_success(self, client: TestClient, db_session):
        """Test successful login."""
        # Create user
        password = "password123"
        user = models.User(
            email="test@example.com",
            hashed_password=get_password_hash(password)
        )
        db_session.add(user)
        db_session.commit()
        
        login_data = {
            "email": "test@example.com",
            "password": password
        }
        
        response = client.post("/api/v1/auth/login", json=login_data)
        
        assert response.status_code == status.HTTP_200_OK
        data = response.json()
        
        assert "access_token" in data
        assert "refresh_token" in data
        assert data["token_type"] == "bearer"
        assert "expires_in" in data
        assert isinstance(data["expires_in"], int)

    def test_login_wrong_password(self, client: TestClient, db_session):
        """Test login with wrong password."""
        user = models.User(
            email="test@example.com",
            hashed_password=get_password_hash("correct_password")
        )
        db_session.add(user)
        db_session.commit()
        
        login_data = {
            "email": "test@example.com",
            "password": "wrong_password"
        }
        
        response = client.post("/api/v1/auth/login", json=login_data)
        
        assert response.status_code == status.HTTP_401_UNAUTHORIZED
        assert "Incorrect email or password" in response.json()["detail"]

    def test_login_user_not_found(self, client: TestClient):
        """Test login with non-existent user."""
        login_data = {
            "email": "notfound@example.com",
            "password": "password123"
        }
        
        response = client.post("/api/v1/auth/login", json=login_data)
        
        assert response.status_code == status.HTTP_401_UNAUTHORIZED
        assert "Incorrect email or password" in response.json()["detail"]

    def test_get_current_user_success(self, client: TestClient, db_session):
        """Test getting current user with valid token."""
        # Create user
        user = models.User(
            email="test@example.com",
            username="testuser",
            full_name="Test User",
            hashed_password="hash"
        )
        db_session.add(user)
        db_session.commit()
        db_session.refresh(user)
        
        # Create token
        token_data = {"sub": str(user.id), "email": user.email}
        access_token = create_access_token(token_data)
        
        headers = {"Authorization": f"Bearer {access_token}"}
        response = client.get("/api/v1/auth/me", headers=headers)
        
        assert response.status_code == status.HTTP_200_OK
        data = response.json()
        
        assert data["email"] == "test@example.com"
        assert data["username"] == "testuser"
        assert data["full_name"] == "Test User"
        assert data["id"] == str(user.id)

    def test_get_current_user_no_token(self, client: TestClient):
        """Test getting current user without token."""
        response = client.get("/api/v1/auth/me")
        
        assert response.status_code == status.HTTP_401_UNAUTHORIZED
        assert "Not authenticated" in response.json()["detail"]

    def test_get_current_user_invalid_token(self, client: TestClient):
        """Test getting current user with invalid token."""
        headers = {"Authorization": "Bearer invalid_token"}
        response = client.get("/api/v1/auth/me", headers=headers)
        
        assert response.status_code == status.HTTP_401_UNAUTHORIZED

    def test_get_current_user_nonexistent_user(self, client: TestClient):
        """Test getting current user with token for non-existent user."""
        # Create token for non-existent user
        token_data = {"sub": str(uuid4()), "email": "notfound@example.com"}
        access_token = create_access_token(token_data)
        
        headers = {"Authorization": f"Bearer {access_token}"}
        response = client.get("/api/v1/auth/me", headers=headers)
        
        assert response.status_code == status.HTTP_401_UNAUTHORIZED

    def test_refresh_token_success(self, client: TestClient, db_session):
        """Test successful token refresh."""
        # Create user
        user = models.User(
            email="test@example.com",
            hashed_password="hash"
        )
        db_session.add(user)
        db_session.commit()
        db_session.refresh(user)
        
        # Create refresh token
        from core.security import create_refresh_token
        token_data = {"sub": str(user.id), "email": user.email}
        refresh_token = create_refresh_token(token_data)
        
        refresh_data = {"refresh_token": refresh_token}
        response = client.post("/api/v1/auth/refresh", json=refresh_data)
        
        assert response.status_code == status.HTTP_200_OK
        data = response.json()
        
        assert "access_token" in data
        assert "refresh_token" in data
        assert data["token_type"] == "bearer"
        assert "expires_in" in data

    def test_refresh_token_invalid(self, client: TestClient):
        """Test token refresh with invalid refresh token."""
        refresh_data = {"refresh_token": "invalid_refresh_token"}
        response = client.post("/api/v1/auth/refresh", json=refresh_data)
        
        assert response.status_code == status.HTTP_401_UNAUTHORIZED

    def test_refresh_token_access_token_used(self, client: TestClient, db_session):
        """Test that access token cannot be used for refresh."""
        # Create user
        user = models.User(
            email="test@example.com",
            hashed_password="hash"
        )
        db_session.add(user)
        db_session.commit()
        db_session.refresh(user)
        
        # Create access token (wrong type)
        token_data = {"sub": str(user.id), "email": user.email}
        access_token = create_access_token(token_data)
        
        refresh_data = {"refresh_token": access_token}
        response = client.post("/api/v1/auth/refresh", json=refresh_data)
        
        assert response.status_code == status.HTTP_401_UNAUTHORIZED

    def test_logout_success(self, client: TestClient, db_session):
        """Test successful logout."""
        # Create user and token
        user = models.User(
            email="test@example.com",
            hashed_password="hash"
        )
        db_session.add(user)
        db_session.commit()
        db_session.refresh(user)
        
        token_data = {"sub": str(user.id), "email": user.email}
        access_token = create_access_token(token_data)
        
        headers = {"Authorization": f"Bearer {access_token}"}
        response = client.post("/api/v1/auth/logout", headers=headers)
        
        assert response.status_code == status.HTTP_200_OK
        data = response.json()
        assert "Successfully logged out" in data["message"]

    def test_logout_no_token(self, client: TestClient):
        """Test logout without token."""
        response = client.post("/api/v1/auth/logout")
        
        assert response.status_code == status.HTTP_401_UNAUTHORIZED

    def test_protected_example_endpoint(self, client: TestClient, db_session):
        """Test protected example endpoint with authentication."""
        # Create user and token
        user = models.User(
            email="test@example.com",
            full_name="Test User",
            hashed_password="hash"
        )
        db_session.add(user)
        db_session.commit()
        db_session.refresh(user)
        
        token_data = {"sub": str(user.id), "email": user.email}
        access_token = create_access_token(token_data)
        
        headers = {"Authorization": f"Bearer {access_token}"}
        response = client.get("/api/v1/auth/protected-example", headers=headers)
        
        assert response.status_code == status.HTTP_200_OK
        data = response.json()
        
        assert "Hello Test User!" in data["message"]
        assert data["user_id"] == str(user.id)
        assert data["authenticated"] is True

    def test_protected_example_endpoint_no_auth(self, client: TestClient):
        """Test protected example endpoint without authentication."""
        response = client.get("/api/v1/auth/protected-example")
        
        assert response.status_code == status.HTTP_401_UNAUTHORIZED

    def test_optional_auth_example_with_auth(self, client: TestClient, db_session):
        """Test optional auth endpoint with authentication."""
        # Create user and token
        user = models.User(
            email="test@example.com",
            full_name="Test User",
            hashed_password="hash"
        )
        db_session.add(user)
        db_session.commit()
        db_session.refresh(user)
        
        token_data = {"sub": str(user.id), "email": user.email}
        access_token = create_access_token(token_data)
        
        headers = {"Authorization": f"Bearer {access_token}"}
        response = client.get("/api/v1/auth/optional-auth-example", headers=headers)
        
        assert response.status_code == status.HTTP_200_OK
        data = response.json()
        
        assert "Welcome back, Test User!" in data["message"]
        assert data["authenticated"] is True

    def test_optional_auth_example_without_auth(self, client: TestClient):
        """Test optional auth endpoint without authentication."""
        response = client.get("/api/v1/auth/optional-auth-example")
        
        assert response.status_code == status.HTTP_200_OK
        data = response.json()
        
        assert "Welcome, anonymous user!" in data["message"]
        assert data["authenticated"] is False

    def test_google_oauth_not_configured(self, client: TestClient):
        """Test Google OAuth when not configured."""
        response = client.get("/api/v1/auth/google")
        
        assert response.status_code == status.HTTP_501_NOT_IMPLEMENTED
        assert "Google OAuth not configured" in response.json()["detail"]

    def test_google_callback_not_configured(self, client: TestClient):
        """Test Google OAuth callback when not configured."""
        response = client.get("/api/v1/auth/google/callback?code=test_code&state=test_state")
        
        assert response.status_code == status.HTTP_501_NOT_IMPLEMENTED
        assert "Google OAuth not configured" in response.json()["detail"]


class TestAuthEndpointsWithInactiveUser:
    """Test authentication endpoints with inactive users."""

    def setup_method(self):
        """Set up test environment."""
        os.environ["JWT_SECRET_KEY"] = "test-secret-key-32-characters-minimum-length"

    def test_get_current_user_inactive_user(self, client: TestClient, db_session):
        """Test that inactive users cannot access protected endpoints."""
        # Create inactive user
        user = models.User(
            email="inactive@example.com",
            hashed_password="hash",
            is_active=False  # Inactive user
        )
        db_session.add(user)
        db_session.commit()
        db_session.refresh(user)
        
        # Create token for inactive user
        token_data = {"sub": str(user.id), "email": user.email}
        access_token = create_access_token(token_data)
        
        headers = {"Authorization": f"Bearer {access_token}"}
        response = client.get("/api/v1/auth/me", headers=headers)
        
        assert response.status_code == status.HTTP_400_BAD_REQUEST
        assert "Inactive user" in response.json()["detail"]


class TestAuthMiddleware:
    """Test authentication middleware functionality."""

    def setup_method(self):
        """Set up test environment."""
        os.environ["JWT_SECRET_KEY"] = "test-secret-key-32-characters-minimum-length"

    def test_bearer_token_extraction(self, client: TestClient, db_session):
        """Test that Bearer tokens are properly extracted."""
        # Create user
        user = models.User(
            email="test@example.com",
            hashed_password="hash"
        )
        db_session.add(user)
        db_session.commit()
        db_session.refresh(user)
        
        token_data = {"sub": str(user.id), "email": user.email}
        access_token = create_access_token(token_data)
        
        # Test with proper Bearer format
        headers = {"Authorization": f"Bearer {access_token}"}
        response = client.get("/api/v1/auth/me", headers=headers)
        assert response.status_code == status.HTTP_200_OK
        
        # Test with missing Bearer prefix (should fail)
        headers = {"Authorization": access_token}
        response = client.get("/api/v1/auth/me", headers=headers)
        assert response.status_code == status.HTTP_401_UNAUTHORIZED

    def test_malformed_authorization_header(self, client: TestClient):
        """Test handling of malformed authorization headers."""
        headers = {"Authorization": "NotBearer token"}
        response = client.get("/api/v1/auth/me", headers=headers)
        
        assert response.status_code == status.HTTP_401_UNAUTHORIZED

    def test_empty_authorization_header(self, client: TestClient):
        """Test handling of empty authorization header."""
        headers = {"Authorization": ""}
        response = client.get("/api/v1/auth/me", headers=headers)
        
        assert response.status_code == status.HTTP_401_UNAUTHORIZED