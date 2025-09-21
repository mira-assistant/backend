"""Test authentication endpoints."""

import pytest
from fastapi.testclient import TestClient
from unittest.mock import Mock
from datetime import datetime, timezone

# Mock the database and dependencies for testing
def mock_get_db():
    """Mock database dependency."""
    pass

def test_auth_endpoints_structure():
    """Test that auth endpoints are properly structured."""
    from api.v1.auth_router import router
    
    # Check that the router has the expected routes
    routes = [route.path for route in router.routes]
    
    expected_routes = [
        "/register",
        "/login", 
        "/refresh",
        "/logout",
        "/me",
        "/google",
        "/google/callback"
    ]
    
    for expected_route in expected_routes:
        assert expected_route in routes

def test_jwt_utilities():
    """Test JWT token creation and verification."""
    from core.security import create_access_token, create_refresh_token, verify_token
    import os
    
    # Set a test secret key
    os.environ["JWT_SECRET_KEY"] = "test-secret-key-32-characters-min"
    
    # Test data
    test_data = {"sub": "test-user-id", "email": "test@example.com"}
    
    # Test access token
    access_token = create_access_token(test_data)
    assert access_token is not None
    assert isinstance(access_token, str)
    
    # Verify access token
    decoded_access = verify_token(access_token, "access")
    assert decoded_access is not None
    assert decoded_access["sub"] == "test-user-id"
    assert decoded_access["email"] == "test@example.com"
    assert decoded_access["type"] == "access"
    
    # Test refresh token
    refresh_token = create_refresh_token(test_data)
    assert refresh_token is not None
    assert isinstance(refresh_token, str)
    
    # Verify refresh token
    decoded_refresh = verify_token(refresh_token, "refresh") 
    assert decoded_refresh is not None
    assert decoded_refresh["sub"] == "test-user-id"
    assert decoded_refresh["type"] == "refresh"
    
    # Test invalid token type verification
    assert verify_token(access_token, "refresh") is None
    assert verify_token(refresh_token, "access") is None

def test_password_hashing():
    """Test password hashing and verification."""
    from core.security import get_password_hash, verify_password
    
    password = "test_password_123"
    hashed = get_password_hash(password)
    
    assert hashed is not None
    assert hashed != password  # Password should be hashed
    assert verify_password(password, hashed) is True
    assert verify_password("wrong_password", hashed) is False

def test_user_model():
    """Test User model structure."""
    from models.user import User
    import uuid
    
    # Test that User model has required fields
    user = User(
        id=uuid.uuid4(),
        email="test@example.com",
        username="testuser",
        full_name="Test User",
        hashed_password="hashed_password_here",
        is_active=True,
        is_verified=False,
        created_at=datetime.now(timezone.utc),
        updated_at=datetime.now(timezone.utc)
    )
    
    assert user.email == "test@example.com"
    assert user.username == "testuser" 
    assert user.full_name == "Test User"
    assert user.is_active is True
    assert user.is_verified is False
    assert user.hashed_password == "hashed_password_here"

def test_auth_schemas():
    """Test authentication Pydantic schemas."""
    from schemas.auth import UserCreate, UserPublic, LoginRequest, Token
    from uuid import uuid4
    from datetime import datetime, timezone
    
    # Test UserCreate schema
    user_create_data = {
        "email": "test@example.com",
        "password": "password123",
        "username": "testuser",
        "full_name": "Test User",
        "is_active": True
    }
    user_create = UserCreate(**user_create_data)
    assert user_create.email == "test@example.com"
    assert user_create.password == "password123"
    
    # Test LoginRequest schema
    login_data = {
        "email": "test@example.com", 
        "password": "password123"
    }
    login_request = LoginRequest(**login_data)
    assert login_request.email == "test@example.com"
    assert login_request.password == "password123"
    
    # Test Token schema
    token_data = {
        "access_token": "access_token_here",
        "refresh_token": "refresh_token_here", 
        "token_type": "bearer",
        "expires_in": 1800
    }
    token = Token(**token_data)
    assert token.access_token == "access_token_here"
    assert token.token_type == "bearer"
    
    # Test UserPublic schema
    user_public_data = {
        "id": uuid4(),
        "email": "test@example.com",
        "username": "testuser",
        "full_name": "Test User",
        "is_active": True,
        "is_verified": False,
        "avatar_url": None,
        "created_at": datetime.now(timezone.utc),
        "last_login": None
    }
    user_public = UserPublic(**user_public_data)
    assert user_public.email == "test@example.com"
    assert user_public.is_active is True

if __name__ == "__main__":
    # Run basic tests
    test_auth_endpoints_structure()
    test_jwt_utilities() 
    test_password_hashing()
    test_user_model()
    test_auth_schemas()
    print("All authentication tests passed!")