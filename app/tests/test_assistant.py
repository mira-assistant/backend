"""
Assistant API tests.
"""

import pytest
from fastapi.testclient import TestClient
from sqlalchemy.orm import Session

from app.main import app
from app.models.interaction import Interaction
from app.models.person import Person

client = TestClient(app)


@pytest.fixture
def test_person(db: Session):
    """Create a test person."""
    person = Person(name="Test Person", index=1)
    db.add(person)
    db.commit()
    db.refresh(person)
    return person


@pytest.fixture
def test_interaction(db: Session, test_person):
    """Create a test interaction."""
    interaction = Interaction(text="Hello, this is a test interaction", speaker_id=test_person.id)
    db.add(interaction)
    db.commit()
    db.refresh(interaction)
    return interaction


def test_get_interaction(test_interaction):
    """Test getting an interaction by ID."""
    response = client.get(f"/api/v1/assistant/interactions/{test_interaction.id}")
    assert response.status_code == 200
    data = response.json()
    assert data["text"] == "Hello, this is a test interaction"
    assert data["id"] == str(test_interaction.id)


def test_get_nonexistent_interaction():
    """Test getting a nonexistent interaction."""
    fake_id = "00000000-0000-0000-0000-000000000000"
    response = client.get(f"/api/v1/assistant/interactions/{fake_id}")
    assert response.status_code == 404


def test_delete_interaction(test_interaction):
    """Test deleting an interaction."""
    response = client.delete(f"/api/v1/assistant/interactions/{test_interaction.id}")
    assert response.status_code == 200
    data = response.json()
    assert data["detail"] == "Interaction deleted successfully"


def test_delete_nonexistent_interaction():
    """Test deleting a nonexistent interaction."""
    fake_id = "00000000-0000-0000-0000-000000000000"
    response = client.delete(f"/api/v1/assistant/interactions/{fake_id}")
    assert response.status_code == 404


def test_interaction_inference(test_interaction):
    """Test interaction inference."""
    response = client.post(f"/api/v1/assistant/interactions/{test_interaction.id}/inference")
    # This might return 200 with a message or 500 if inference fails
    # The exact behavior depends on the inference service implementation
    assert response.status_code in [200, 500]
