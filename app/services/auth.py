"""Authentication service layer."""

from datetime import datetime, timezone
from typing import Optional
from uuid import UUID

from fastapi import HTTPException, status
from sqlalchemy.orm import Session

import models
from core.security import get_password_hash, verify_password
from schemas.auth import UserCreate, UserInDB


class AuthService:
    """Service class for authentication operations."""

    @staticmethod
    def get_user_by_email(db: Session, email: str) -> Optional[models.User]:
        """Get user by email."""
        return db.query(models.User).filter(models.User.email == email).first()

    @staticmethod
    def get_user_by_id(db: Session, user_id: UUID) -> Optional[models.User]:
        """Get user by ID."""
        return db.query(models.User).filter(models.User.id == user_id).first()

    @staticmethod
    def get_user_by_google_id(db: Session, google_id: str) -> Optional[models.User]:
        """Get user by Google ID."""
        return db.query(models.User).filter(models.User.google_id == google_id).first()

    @staticmethod
    def create_user(db: Session, user_create: UserCreate) -> models.User:
        """Create a new user."""
        # Check if user already exists
        if AuthService.get_user_by_email(db, user_create.email):
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Email already registered"
            )

        # Create new user
        hashed_password = get_password_hash(user_create.password)
        db_user = models.User(
            email=user_create.email,
            username=user_create.username,
            full_name=user_create.full_name,
            hashed_password=hashed_password,
            is_active=user_create.is_active,
        )
        
        db.add(db_user)
        db.commit()
        db.refresh(db_user)
        return db_user

    @staticmethod
    def authenticate_user(db: Session, email: str, password: str) -> Optional[models.User]:
        """Authenticate user with email and password."""
        user = AuthService.get_user_by_email(db, email)
        if not user:
            return None
        if not user.hashed_password:
            return None  # OAuth user without password
        if not verify_password(password, user.hashed_password):
            return None
        
        # Update last login
        user.last_login = datetime.now(timezone.utc)
        db.commit()
        
        return user

    @staticmethod
    def create_or_update_google_user(
        db: Session, 
        google_id: str, 
        email: str, 
        full_name: Optional[str] = None,
        avatar_url: Optional[str] = None
    ) -> models.User:
        """Create or update user from Google OAuth."""
        # Check if user exists by Google ID
        user = AuthService.get_user_by_google_id(db, google_id)
        
        if user:
            # Update existing user
            user.full_name = full_name or user.full_name
            user.avatar_url = avatar_url or user.avatar_url
            user.last_login = datetime.now(timezone.utc)
            user.is_verified = True  # Google users are verified
        else:
            # Check if user exists by email
            user = AuthService.get_user_by_email(db, email)
            if user:
                # Link Google account to existing user
                user.google_id = google_id
                user.avatar_url = avatar_url or user.avatar_url
                user.is_verified = True
                user.last_login = datetime.now(timezone.utc)
            else:
                # Create new user
                user = models.User(
                    email=email,
                    full_name=full_name,
                    google_id=google_id,
                    avatar_url=avatar_url,
                    is_active=True,
                    is_verified=True,
                    last_login=datetime.now(timezone.utc)
                )
                db.add(user)
        
        db.commit()
        db.refresh(user)
        return user

    @staticmethod
    def update_user_login(db: Session, user: models.User) -> None:
        """Update user's last login time."""
        user.last_login = datetime.now(timezone.utc)
        db.commit()