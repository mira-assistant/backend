"""
Simple test for core authentication utilities.
"""

from datetime import datetime, timedelta, timezone
from core.auth import create_access_token, create_refresh_token, verify_token, get_password_hash, verify_password


def test_password_hashing():
    """Test password hashing and verification."""
    password = "test_password_123"
    hashed = get_password_hash(password)
    
    # Verify password
    assert verify_password(password, hashed)
    
    # Wrong password should fail
    assert not verify_password("wrong_password", hashed)


def test_jwt_tokens():
    """Test JWT token creation and verification."""
    user_data = {"sub": "test-user-id"}
    
    # Create access token
    access_token = create_access_token(user_data)
    assert access_token is not None
    
    # Verify access token
    payload = verify_token(access_token, "access")
    assert payload is not None
    assert payload["sub"] == "test-user-id"
    assert payload["type"] == "access"
    
    # Create refresh token
    refresh_token = create_refresh_token(user_data)
    assert refresh_token is not None
    
    # Verify refresh token
    payload = verify_token(refresh_token, "refresh")
    assert payload is not None
    assert payload["sub"] == "test-user-id"
    assert payload["type"] == "refresh"
    
    # Wrong token type should fail
    assert verify_token(access_token, "refresh") is None
    assert verify_token(refresh_token, "access") is None


def test_invalid_token():
    """Test invalid token verification."""
    assert verify_token("invalid_token") is None
    assert verify_token("") is None


if __name__ == "__main__":
    test_password_hashing()
    test_jwt_tokens()
    test_invalid_token()
    print("All core auth tests passed!")