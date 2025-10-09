"""
Tests for authentication endpoints.
"""

from fastapi.testclient import TestClient

from app.main import app


class TestAuth:
    """Test cases for authentication endpoints."""

    def test_auth_endpoints_exist(self):
        """Test that auth endpoints are registered."""
        client = TestClient(app)

        # Test login endpoint
        response = client.post(
            "/api/v2/auth/login", json={"username": "test", "password": "test"}
        )
        # Should get 422 for invalid data, not 404 for missing endpoint
        assert response.status_code != 404

        # Test Google OAuth URL endpoint (login endpoint is commented out)
        response = client.get("/api/v2/auth/google/url")
        assert response.status_code != 404

        # Test GitHub OAuth endpoints (returns 302 redirect, not 404)
        response = client.get("/api/v2/auth/github/login", follow_redirects=False)
        assert response.status_code != 404

        # Test refresh endpoint
        response = client.post(
            "/api/v2/auth/refresh", json={"refresh_token": "invalid"}
        )
        assert response.status_code != 404

    def test_login_invalid_credentials(self):
        """Test login with invalid credentials."""
        client = TestClient(app)

        response = client.post(
            "/api/v2/auth/login",
            json={"username": "nonexistent", "password": "wrongpassword"},
        )

        # Should get 401 for invalid credentials
        assert response.status_code == 401
        assert "Incorrect username or password" in response.json()["detail"]

    def test_refresh_invalid_token(self):
        """Test refresh with invalid token."""
        client = TestClient(app)

        response = client.post(
            "/api/v2/auth/refresh", json={"refresh_token": "invalid_token"}
        )

        # Should get 401 for invalid token
        assert response.status_code == 401
        assert "Invalid or expired refresh token" in response.json()["detail"]
