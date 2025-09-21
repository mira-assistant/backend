"""Unit tests for authentication system."""

import os
import pytest
from datetime import datetime, timezone
from uuid import uuid4
from unittest.mock import patch

from fastapi import HTTPException, status
from sqlalchemy.orm import Session

import models
from core.security import (
    create_access_token,
    create_refresh_token,
    verify_token,
    get_password_hash,
    verify_password,
    get_token_expire_time
)
from schemas.auth import UserCreate, UserPublic, LoginRequest, Token
from services.auth import AuthService


class TestJWTUtilities:
    """Test JWT token creation and verification."""

    def setup_method(self):
        """Set up test environment."""
        os.environ["JWT_SECRET_KEY"] = "test-secret-key-32-characters-minimum-length"

    def test_create_access_token(self):
        """Test access token creation."""
        test_data = {"sub": "test-user-id", "email": "test@example.com"}
        token = create_access_token(test_data)
        
        assert token is not None
        assert isinstance(token, str)
        assert len(token) > 20  # JWT tokens are typically much longer

    def test_create_refresh_token(self):
        """Test refresh token creation."""
        test_data = {"sub": "test-user-id", "email": "test@example.com"}
        token = create_refresh_token(test_data)
        
        assert token is not None
        assert isinstance(token, str)
        assert len(token) > 20

    def test_verify_access_token(self):
        """Test access token verification."""
        test_data = {"sub": "test-user-id", "email": "test@example.com"}
        token = create_access_token(test_data)
        
        decoded = verify_token(token, "access")
        assert decoded is not None
        assert decoded["sub"] == "test-user-id"
        assert decoded["email"] == "test@example.com"
        assert decoded["type"] == "access"

    def test_verify_refresh_token(self):
        """Test refresh token verification."""
        test_data = {"sub": "test-user-id", "email": "test@example.com"}
        token = create_refresh_token(test_data)
        
        decoded = verify_token(token, "refresh")
        assert decoded is not None
        assert decoded["sub"] == "test-user-id"
        assert decoded["type"] == "refresh"

    def test_token_type_mismatch(self):
        """Test that wrong token type returns None."""
        test_data = {"sub": "test-user-id", "email": "test@example.com"}
        access_token = create_access_token(test_data)
        refresh_token = create_refresh_token(test_data)
        
        # Access token should not verify as refresh token
        assert verify_token(access_token, "refresh") is None
        # Refresh token should not verify as access token
        assert verify_token(refresh_token, "access") is None

    def test_invalid_token(self):
        """Test verification of invalid token."""
        invalid_token = "invalid.token.here"
        assert verify_token(invalid_token, "access") is None

    def test_get_token_expire_time(self):
        """Test token expire time calculation."""
        expire_time = get_token_expire_time()
        assert isinstance(expire_time, int)
        assert expire_time > 0


class TestPasswordHashing:
    """Test password hashing and verification."""

    def test_password_hashing(self):
        """Test password hashing."""
        password = "test_password_123"
        hashed = get_password_hash(password)
        
        assert hashed is not None
        assert hashed != password  # Password should be hashed
        assert len(hashed) > 20  # Bcrypt hashes are typically 60 characters

    def test_password_verification(self):
        """Test password verification."""
        password = "test_password_123"
        hashed = get_password_hash(password)
        
        assert verify_password(password, hashed) is True
        assert verify_password("wrong_password", hashed) is False

    def test_different_passwords_different_hashes(self):
        """Test that same password produces different hashes (salt)."""
        password = "test_password_123"
        hash1 = get_password_hash(password)
        hash2 = get_password_hash(password)
        
        # Hashes should be different due to salt
        assert hash1 != hash2
        # But both should verify correctly
        assert verify_password(password, hash1) is True
        assert verify_password(password, hash2) is True


class TestAuthSchemas:
    """Test authentication Pydantic schemas."""

    def test_user_create_schema(self):
        """Test UserCreate schema validation."""
        user_data = {
            "email": "test@example.com",
            "password": "password123",
            "username": "testuser",
            "full_name": "Test User",
            "is_active": True
        }
        user_create = UserCreate(**user_data)
        
        assert user_create.email == "test@example.com"
        assert user_create.password == "password123"
        assert user_create.username == "testuser"
        assert user_create.full_name == "Test User"
        assert user_create.is_active is True

    def test_user_create_password_validation(self):
        """Test password length validation."""
        with pytest.raises(ValueError):
            UserCreate(
                email="test@example.com",
                password="short"  # Too short
            )

    def test_user_create_email_validation(self):
        """Test email validation."""
        with pytest.raises(ValueError):
            UserCreate(
                email="invalid-email",
                password="password123"
            )

    def test_login_request_schema(self):
        """Test LoginRequest schema."""
        login_data = {
            "email": "test@example.com",
            "password": "password123"
        }
        login_request = LoginRequest(**login_data)
        
        assert login_request.email == "test@example.com"
        assert login_request.password == "password123"

    def test_token_schema(self):
        """Test Token schema."""
        token_data = {
            "access_token": "access_token_here",
            "refresh_token": "refresh_token_here",
            "token_type": "bearer",
            "expires_in": 1800
        }
        token = Token(**token_data)
        
        assert token.access_token == "access_token_here"
        assert token.refresh_token == "refresh_token_here"
        assert token.token_type == "bearer"
        assert token.expires_in == 1800

    def test_user_public_schema(self):
        """Test UserPublic schema."""
        user_data = {
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
        user_public = UserPublic(**user_data)
        
        assert user_public.email == "test@example.com"
        assert user_public.username == "testuser"
        assert user_public.is_active is True
        assert user_public.is_verified is False


class TestUserModel:
    """Test User model."""

    def test_user_model_creation(self, db_session: Session):
        """Test creating a User instance."""
        user = models.User(
            email="test@example.com",
            username="testuser",
            full_name="Test User",
            hashed_password="hashed_password_here",
            is_active=True,
            is_verified=False
        )
        
        db_session.add(user)
        db_session.commit()
        db_session.refresh(user)
        
        assert user.id is not None
        assert user.email == "test@example.com"
        assert user.username == "testuser"
        assert user.full_name == "Test User"
        assert user.hashed_password == "hashed_password_here"
        assert user.is_active is True
        assert user.is_verified is False
        assert user.created_at is not None
        assert user.updated_at is not None

    def test_user_email_unique_constraint(self, db_session: Session):
        """Test that email must be unique."""
        user1 = models.User(
            email="test@example.com",
            hashed_password="hash1"
        )
        user2 = models.User(
            email="test@example.com",  # Same email
            hashed_password="hash2"
        )
        
        db_session.add(user1)
        db_session.commit()
        
        db_session.add(user2)
        with pytest.raises(Exception):  # Should raise integrity error
            db_session.commit()

    def test_user_google_id_unique_constraint(self, db_session: Session):
        """Test that google_id must be unique."""
        user1 = models.User(
            email="test1@example.com",
            google_id="google_id_123"
        )
        user2 = models.User(
            email="test2@example.com",
            google_id="google_id_123"  # Same Google ID
        )
        
        db_session.add(user1)
        db_session.commit()
        
        db_session.add(user2)
        with pytest.raises(Exception):  # Should raise integrity error
            db_session.commit()

    def test_user_oauth_fields(self, db_session: Session):
        """Test OAuth-specific fields."""
        user = models.User(
            email="oauth@example.com",
            google_id="google_123",
            avatar_url="https://example.com/avatar.jpg",
            is_verified=True,
            hashed_password=None  # OAuth users may not have password
        )
        
        db_session.add(user)
        db_session.commit()
        db_session.refresh(user)
        
        assert user.google_id == "google_123"
        assert user.avatar_url == "https://example.com/avatar.jpg"
        assert user.is_verified is True
        assert user.hashed_password is None


class TestAuthService:
    """Test authentication service methods."""

    def test_get_user_by_email(self, db_session: Session):
        """Test getting user by email."""
        user = models.User(
            email="test@example.com",
            hashed_password="hashed_password"
        )
        db_session.add(user)
        db_session.commit()
        
        found_user = AuthService.get_user_by_email(db_session, "test@example.com")
        assert found_user is not None
        assert found_user.email == "test@example.com"
        
        not_found = AuthService.get_user_by_email(db_session, "notfound@example.com")
        assert not_found is None

    def test_get_user_by_id(self, db_session: Session):
        """Test getting user by ID."""
        user = models.User(
            email="test@example.com",
            hashed_password="hashed_password"
        )
        db_session.add(user)
        db_session.commit()
        db_session.refresh(user)
        
        found_user = AuthService.get_user_by_id(db_session, user.id)
        assert found_user is not None
        assert found_user.id == user.id
        
        not_found = AuthService.get_user_by_id(db_session, uuid4())
        assert not_found is None

    def test_get_user_by_google_id(self, db_session: Session):
        """Test getting user by Google ID."""
        user = models.User(
            email="test@example.com",
            google_id="google_123"
        )
        db_session.add(user)
        db_session.commit()
        
        found_user = AuthService.get_user_by_google_id(db_session, "google_123")
        assert found_user is not None
        assert found_user.google_id == "google_123"
        
        not_found = AuthService.get_user_by_google_id(db_session, "not_found")
        assert not_found is None

    def test_create_user(self, db_session: Session):
        """Test creating a new user."""
        user_create = UserCreate(
            email="test@example.com",
            password="password123",
            username="testuser",
            full_name="Test User"
        )
        
        user = AuthService.create_user(db_session, user_create)
        
        assert user.email == "test@example.com"
        assert user.username == "testuser"
        assert user.full_name == "Test User"
        assert user.hashed_password is not None
        assert user.hashed_password != "password123"  # Should be hashed
        assert verify_password("password123", user.hashed_password) is True

    def test_create_user_duplicate_email(self, db_session: Session):
        """Test creating user with duplicate email raises error."""
        # Create first user
        user1 = models.User(
            email="test@example.com",
            hashed_password="hash1"
        )
        db_session.add(user1)
        db_session.commit()
        
        # Try to create second user with same email
        user_create = UserCreate(
            email="test@example.com",
            password="password123"
        )
        
        with pytest.raises(HTTPException) as exc_info:
            AuthService.create_user(db_session, user_create)
        
        assert exc_info.value.status_code == status.HTTP_400_BAD_REQUEST
        assert "Email already registered" in str(exc_info.value.detail)

    def test_authenticate_user_success(self, db_session: Session):
        """Test successful user authentication."""
        # Create user with hashed password
        password = "password123"
        user = models.User(
            email="test@example.com",
            hashed_password=get_password_hash(password)
        )
        db_session.add(user)
        db_session.commit()
        
        authenticated_user = AuthService.authenticate_user(
            db_session, "test@example.com", password
        )
        
        assert authenticated_user is not None
        assert authenticated_user.email == "test@example.com"
        assert authenticated_user.last_login is not None

    def test_authenticate_user_wrong_password(self, db_session: Session):
        """Test authentication with wrong password."""
        user = models.User(
            email="test@example.com",
            hashed_password=get_password_hash("correct_password")
        )
        db_session.add(user)
        db_session.commit()
        
        authenticated_user = AuthService.authenticate_user(
            db_session, "test@example.com", "wrong_password"
        )
        
        assert authenticated_user is None

    def test_authenticate_user_not_found(self, db_session: Session):
        """Test authentication with non-existent user."""
        authenticated_user = AuthService.authenticate_user(
            db_session, "notfound@example.com", "password123"
        )
        
        assert authenticated_user is None

    def test_authenticate_oauth_user_no_password(self, db_session: Session):
        """Test authentication fails for OAuth user without password."""
        user = models.User(
            email="oauth@example.com",
            google_id="google_123",
            hashed_password=None  # OAuth user without password
        )
        db_session.add(user)
        db_session.commit()
        
        authenticated_user = AuthService.authenticate_user(
            db_session, "oauth@example.com", "any_password"
        )
        
        assert authenticated_user is None

    def test_create_or_update_google_user_new(self, db_session: Session):
        """Test creating new Google user."""
        user = AuthService.create_or_update_google_user(
            db=db_session,
            google_id="google_123",
            email="newuser@gmail.com",
            full_name="New User",
            avatar_url="https://example.com/avatar.jpg"
        )
        
        assert user.email == "newuser@gmail.com"
        assert user.google_id == "google_123"
        assert user.full_name == "New User"
        assert user.avatar_url == "https://example.com/avatar.jpg"
        assert user.is_verified is True
        assert user.last_login is not None

    def test_create_or_update_google_user_existing_google_id(self, db_session: Session):
        """Test updating existing Google user."""
        # Create existing user
        existing_user = models.User(
            email="existing@gmail.com",
            google_id="google_123",
            full_name="Old Name"
        )
        db_session.add(existing_user)
        db_session.commit()
        
        # Update user
        updated_user = AuthService.create_or_update_google_user(
            db=db_session,
            google_id="google_123",
            email="existing@gmail.com",
            full_name="New Name",
            avatar_url="https://example.com/new_avatar.jpg"
        )
        
        assert updated_user.id == existing_user.id
        assert updated_user.full_name == "New Name"
        assert updated_user.avatar_url == "https://example.com/new_avatar.jpg"
        assert updated_user.last_login is not None

    def test_create_or_update_google_user_existing_email(self, db_session: Session):
        """Test linking Google account to existing email user."""
        # Create existing user with email but no Google ID
        existing_user = models.User(
            email="existing@gmail.com",
            hashed_password="old_password_hash"
        )
        db_session.add(existing_user)
        db_session.commit()
        
        # Link Google account
        updated_user = AuthService.create_or_update_google_user(
            db=db_session,
            google_id="google_123",
            email="existing@gmail.com",
            full_name="Full Name"
        )
        
        assert updated_user.id == existing_user.id
        assert updated_user.google_id == "google_123"
        assert updated_user.is_verified is True
        assert updated_user.hashed_password == "old_password_hash"  # Keep existing password

    def test_update_user_login(self, db_session: Session):
        """Test updating user login time."""
        user = models.User(
            email="test@example.com",
            hashed_password="hash"
        )
        db_session.add(user)
        db_session.commit()
        
        assert user.last_login is None
        
        AuthService.update_user_login(db_session, user)
        
        assert user.last_login is not None
        assert isinstance(user.last_login, datetime)