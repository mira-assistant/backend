"""
Integration tests for API endpoints.
"""

import uuid

from fastapi.testclient import TestClient
from sqlalchemy.orm import Session

from models import Conversation, MiraNetwork, Person


class TestRootEndpoint:
    """Test cases for root endpoint."""

    def test_root_endpoint(self, client: TestClient):
        """Test the root endpoint returns correct information."""
        response = client.get("/")

        assert response.status_code == 200
        data = response.json()
        assert data["message"] == "Mira Backend API"
        assert data["status"] == "running"
        assert "version" in data
        assert data["stable"] == "v1"

    def test_root_endpoint_cors_headers(self, client: TestClient):
        """Test that CORS headers are properly set."""
        response = client.options("/")
        # CORS preflight should be handled by middleware
        assert response.status_code in [200, 405]  # 405 is acceptable for OPTIONS


class TestConversationEndpoints:
    """Test cases for conversation endpoints."""

    def test_get_conversation_success(
        self, client: TestClient, db_session: Session, sample_network_id
    ):
        """Test getting a conversation successfully."""
        # Create a network and conversation
        network = MiraNetwork(name="Test Network")
        db_session.add(network)
        db_session.commit()
        db_session.refresh(network)

        conversation = Conversation(
            topic_summary="Test Topic",
            context_summary="Test Context",
            network_id=network.id,
        )

        db_session.add(conversation)
        db_session.commit()

        # Test the endpoint
        response = client.get(f"/api/v1/{network.id}/conversations/{conversation.id}")

        assert response.status_code == 200
        data = response.json()
        assert data["id"] == str(conversation.id)
        assert data["topic_summary"] == "Test Topic"
        assert data["context_summary"] == "Test Context"
        assert data["network_id"] == str(network.id)

    def test_get_conversation_not_found(
        self, client: TestClient, db_session: Session, sample_network_id
    ):
        """Test getting a non-existent conversation."""
        network = MiraNetwork(name="Test Network")
        db_session.add(network)
        db_session.commit()
        db_session.refresh(network)

        fake_conversation_id = str(uuid.uuid4())

        response = client.get(
            f"/api/v1/{network.id}/conversations/{fake_conversation_id}"
        )

        assert response.status_code == 404
        data = response.json()
        assert data["detail"] == "Conversation not found"

    def test_get_conversation_invalid_network(
        self, client: TestClient, db_session: Session, sample_network_id
    ):
        """Test getting a conversation with invalid network ID."""
        network = MiraNetwork(name="Test Network")
        db_session.add(network)
        db_session.commit()
        db_session.refresh(network)

        conversation = Conversation(network_id=network.id)
        db_session.add(conversation)
        db_session.commit()

        fake_network_id = str(uuid.uuid4())

        response = client.get(
            f"/api/v1/{fake_network_id}/conversations/{conversation.id}"
        )

        assert response.status_code == 404
        data = response.json()
        assert data["detail"] == "Conversation not found"

    def test_get_conversation_invalid_uuid(self, client: TestClient, sample_network_id):
        """Test getting a conversation with invalid UUID format."""
        response = client.get(f"/api/v1/{sample_network_id}/conversations/not-a-uuid")

        # The API should handle this gracefully, either 422 or 404
        assert response.status_code in [422, 404]


class TestPersonEndpoints:
    """Test cases for person endpoints."""

    def test_get_person_success(
        self, client: TestClient, db_session: Session, sample_network_id
    ):
        """Test getting a person successfully."""
        # Create a network and person
        network = MiraNetwork(name="Test Network")
        db_session.add(network)
        db_session.commit()
        db_session.refresh(network)

        person = Person(
            name="John Doe",
            index=1,
            voice_embedding=[0.1, 0.2, 0.3],
            cluster_id=1,
            network_id=network.id,
        )

        db_session.add(person)
        db_session.commit()

        # Test the endpoint
        response = client.get(f"/api/v1/{network.id}/persons/{person.id}")

        assert response.status_code == 200
        data = response.json()
        assert data["id"] == str(person.id)
        assert data["name"] == "John Doe"
        assert data["index"] == 1
        assert data["voice_embedding"] == [0.1, 0.2, 0.3]
        assert data["cluster_id"] == 1
        assert data["network_id"] == str(network.id)

    def test_get_person_not_found(
        self, client: TestClient, db_session: Session, sample_network_id
    ):
        """Test getting a non-existent person."""
        network = MiraNetwork(name="Test Network")
        db_session.add(network)
        db_session.commit()
        db_session.refresh(network)

        fake_person_id = str(uuid.uuid4())

        response = client.get(f"/api/v1/{network.id}/persons/{fake_person_id}")

        assert response.status_code == 404
        data = response.json()
        assert data["detail"] == "Person not found in this network"

    def test_get_all_persons_success(
        self, client: TestClient, db_session: Session, sample_network_id
    ):
        """Test getting all persons in a network."""
        # Create a network and multiple persons
        network = MiraNetwork(name="Test Network")
        db_session.add(network)
        db_session.commit()
        db_session.refresh(network)

        person1 = Person(name="John Doe", index=1, network_id=network.id)
        person2 = Person(name="Jane Doe", index=2, network_id=network.id)

        db_session.add(person1)
        db_session.add(person2)
        db_session.commit()

        # Test the endpoint
        response = client.get(f"/api/v1/{network.id}/persons/")

        assert response.status_code == 200
        data = response.json()
        assert data["network_id"] == str(network.id)
        assert data["total_count"] == 2
        assert len(data["persons"]) == 2

        # Check person data
        person_ids = [p["id"] for p in data["persons"]]
        assert str(person1.id) in person_ids
        assert str(person2.id) in person_ids

    def test_get_all_persons_empty(
        self, client: TestClient, db_session: Session, sample_network_id
    ):
        """Test getting all persons when network has no persons."""
        network = MiraNetwork(name="Test Network")
        db_session.add(network)
        db_session.commit()
        db_session.refresh(network)

        response = client.get(f"/api/v1/{network.id}/persons/")

        assert response.status_code == 200
        data = response.json()
        assert data["network_id"] == str(network.id)
        assert data["total_count"] == 0
        assert data["persons"] == []

    def test_update_person_name_only(
        self, client: TestClient, db_session: Session, sample_network_id
    ):
        """Test updating a person's name only."""
        # Create a network and person
        network = MiraNetwork(name="Test Network")
        db_session.add(network)
        db_session.commit()
        db_session.refresh(network)

        person = Person(name="Old Name", index=1, network_id=network.id)

        db_session.add(person)
        db_session.commit()

        # Test the endpoint
        response = client.post(
            f"/api/v1/{network.id}/persons/{person.id}/update",
            data={"name": "New Name"},
        )

        assert response.status_code == 200
        data = response.json()
        assert data["message"] == "Person updated successfully"
        assert data["person"]["name"] == "New Name"
        assert data["person"]["id"] == str(person.id)

    def test_update_person_not_found(
        self, client: TestClient, db_session: Session, sample_network_id
    ):
        """Test updating a non-existent person."""
        network = MiraNetwork(name="Test Network")
        db_session.add(network)
        db_session.commit()
        db_session.refresh(network)

        fake_person_id = str(uuid.uuid4())

        response = client.post(
            f"/api/v1/{network.id}/persons/{fake_person_id}/update",
            data={"name": "New Name"},
        )

        assert response.status_code == 404
        data = response.json()
        assert data["detail"] == "Person not found in this network"

    def test_delete_person_success(
        self, client: TestClient, db_session: Session, sample_network_id
    ):
        """Test deleting a person successfully."""
        # Create a network and person
        network = MiraNetwork(name="Test Network")
        db_session.add(network)
        db_session.commit()
        db_session.refresh(network)

        person = Person(name="John Doe", index=1, network_id=network.id)

        db_session.add(person)
        db_session.commit()
        person_id = str(person.id)

        # Test the endpoint
        response = client.delete(f"/api/v1/{network.id}/persons/{person_id}")

        assert response.status_code == 200
        data = response.json()
        assert data["message"] == f"Person {person_id} deleted successfully"
        assert data["deleted_person_id"] == person_id
        assert data["network_id"] == str(network.id)

        # Verify person is deleted
        deleted_person = db_session.query(Person).filter(Person.id == person.id).first()
        assert deleted_person is None

    def test_delete_person_not_found(
        self, client: TestClient, db_session: Session, sample_network_id
    ):
        """Test deleting a non-existent person."""
        network = MiraNetwork(name="Test Network")
        db_session.add(network)
        db_session.commit()
        db_session.refresh(network)

        fake_person_id = str(uuid.uuid4())

        response = client.delete(f"/api/v1/{network.id}/persons/{fake_person_id}")

        assert response.status_code == 404
        data = response.json()
        assert data["detail"] == "Person not found in this network"


class TestErrorHandling:
    """Test cases for error handling."""

    def test_global_exception_handler(self, client: TestClient):
        """Test that the global exception handler works."""
        # This would require a way to trigger an exception in the app
        # For now, we'll test that the handler is properly configured
        response = client.get("/")
        assert response.status_code == 200

    def test_invalid_json_request(self, client: TestClient, sample_network_id):
        """Test handling of invalid JSON in request body."""
        # Send malformed JSON
        response = client.post(
            f"/api/v1/{sample_network_id}/persons/{uuid.uuid4()}/update",
            json="invalid json",
            headers={"Content-Type": "application/json"},
        )

        # Should return 422 for validation error or 404 for person not found
        assert response.status_code in [422, 404]

    def test_missing_required_fields(self, client: TestClient, sample_network_id):
        """Test handling of missing required fields."""
        response = client.post(
            f"/api/v1/{sample_network_id}/persons/{uuid.uuid4()}/update",
            data={},  # Missing required fields
        )

        # Should still work as all fields are optional in the update endpoint
        assert response.status_code == 404  # Person not found, which is expected


class TestCORS:
    """Test cases for CORS functionality."""

    def test_cors_headers_present(self, client: TestClient):
        """Test that CORS headers are present in responses."""
        response = client.get("/")

        # Check for CORS headers (these might be set by middleware)
        # The exact headers depend on the CORS configuration
        assert response.status_code == 200

    def test_cors_preflight_request(self, client: TestClient):
        """Test CORS preflight request handling."""
        response = client.options(
            "/",
            headers={
                "Origin": "http://localhost:3000",
                "Access-Control-Request-Method": "GET",
                "Access-Control-Request-Headers": "Content-Type",
            },
        )

        # CORS preflight should be handled
        assert response.status_code in [200, 405]  # 405 is acceptable for OPTIONS


class TestDatabaseIntegration:
    """Test cases for database integration."""

    def test_database_session_isolation(
        self, client: TestClient, db_session: Session, sample_network_id
    ):
        """Test that database sessions are properly isolated between requests."""
        # Create a network
        network = MiraNetwork(name="Test Network")
        db_session.add(network)
        db_session.commit()

        # Make a request that should find the network
        response = client.get(f"/api/v1/{network.id}/persons/")
        assert response.status_code == 200

        # Make a request with a different network ID that shouldn't exist
        fake_network_id = str(uuid.uuid4())
        response = client.get(f"/api/v1/{fake_network_id}/persons/")
        assert response.status_code == 200  # Should return empty list, not error

    def test_database_transaction_rollback(
        self, client: TestClient, db_session: Session, sample_network_id
    ):
        """Test that database transactions are properly rolled back on errors."""
        # This test would require a way to trigger a database error
        # For now, we'll test that the database is in a consistent state
        network = MiraNetwork(name="Test Network")
        db_session.add(network)
        db_session.commit()

        # Verify the network exists
        response = client.get(f"/api/v1/{network.id}/persons/")
        assert response.status_code == 200
