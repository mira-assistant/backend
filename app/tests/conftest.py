"""
Pytest configuration and shared fixtures for Mira Backend tests.
"""

import os
import sys
import tempfile
import uuid
from typing import AsyncGenerator, Generator

import pytest
from fastapi.testclient import TestClient
from httpx import AsyncClient
from sqlalchemy import create_engine
from sqlalchemy.orm import Session, sessionmaker

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from app.db import get_db
from app.db.base import Base
from app.main import app


@pytest.fixture(scope="session")
def test_db_url() -> str:
    """Create a temporary SQLite database for testing."""
    # Create a temporary file for the test database
    temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".db")
    temp_file.close()
    return f"sqlite:///{temp_file.name}"


@pytest.fixture(scope="session")
def test_engine(test_db_url: str):
    """Create a test database engine."""
    engine = create_engine(
        test_db_url, connect_args={"check_same_thread": False}, echo=False
    )
    return engine


@pytest.fixture(scope="session")
def test_session_factory(test_engine):
    """Create a test session factory."""
    return sessionmaker(autocommit=False, autoflush=False, bind=test_engine)


@pytest.fixture(scope="function")
def db_session(test_session_factory, test_engine) -> Generator[Session, None, None]:
    """Create a fresh database session for each test."""
    # Create all tables
    Base.metadata.create_all(bind=test_engine)

    # Create a session
    session = test_session_factory()

    try:
        yield session
    finally:
        session.close()
        # Drop all tables after each test
        Base.metadata.drop_all(bind=test_engine)


@pytest.fixture(scope="function")
def client(db_session: Session) -> Generator[TestClient, None, None]:
    """Create a test client with database dependency override."""

    def override_get_db():
        try:
            yield db_session
        finally:
            pass

    app.dependency_overrides[get_db] = override_get_db

    with TestClient(app) as test_client:
        yield test_client

    app.dependency_overrides.clear()


@pytest.fixture(scope="function")
async def async_client(db_session: Session) -> AsyncGenerator[AsyncClient, None]:
    """Create an async test client with database dependency override."""

    def override_get_db():
        try:
            yield db_session
        finally:
            pass

    app.dependency_overrides[get_db] = override_get_db

    async with AsyncClient(base_url="http://test") as ac:
        yield ac

    app.dependency_overrides.clear()


@pytest.fixture
def sample_network_id() -> str:
    """Generate a sample network ID for testing."""

    return str(uuid.uuid4())


@pytest.fixture
def sample_person_id() -> str:
    """Generate a sample person ID for testing."""

    return str(uuid.uuid4())


@pytest.fixture
def sample_conversation_id() -> str:
    """Generate a sample conversation ID for testing."""

    return str(uuid.uuid4())


@pytest.fixture
def sample_interaction_id() -> str:
    """Generate a sample interaction ID for testing."""

    return str(uuid.uuid4())


@pytest.fixture
def sample_action_id() -> str:
    """Generate a sample action ID for testing."""

    return str(uuid.uuid4())


@pytest.fixture
def cleanup_temp_files():
    """Cleanup temporary files after tests."""
    temp_files = []

    def add_temp_file(file_path: str):
        temp_files.append(file_path)

    yield add_temp_file

    # Cleanup
    for file_path in temp_files:
        try:
            if os.path.exists(file_path):
                os.remove(file_path)
        except OSError:
            pass  # File might already be deleted
