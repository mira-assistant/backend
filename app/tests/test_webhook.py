"""
Tests for webhook functionality in the multi-tenant backend.
"""

import json
import uuid
from datetime import datetime, timezone
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi.testclient import TestClient

from app.core.auth import create_access_token
from app.models import Interaction, MiraNetwork, User
from app.schemas.client import WebhookPayload


@pytest.fixture
def sample_user(db_session):
    """Create a sample user for testing."""
    from app.core.auth import get_password_hash
    
    user = User(
        id=uuid.uuid4(),
        username="testuser",
        email="test@example.com",
        password_hash=get_password_hash("testpassword"),
        is_active=True,
    )
    db_session.add(user)
    db_session.commit()
    db_session.refresh(user)
    return user


@pytest.fixture
def auth_headers(sample_user):
    """Create authentication headers with JWT token."""
    access_token = create_access_token(data={"sub": str(sample_user.id)})
    return {"Authorization": f"Bearer {access_token}"}


@pytest.fixture
def sample_network(db_session, sample_user):
    """Create a sample network for testing."""
    network = MiraNetwork(
        id=sample_user.id,
        name="Test Network",
        service_enabled=True,
        connected_clients={},
    )
    db_session.add(network)
    db_session.commit()
    db_session.refresh(network)
    return network


def test_client_registration_with_webhook(client, auth_headers):
    """Test client registration with webhook URL."""
    client_id = "test-client-1"
    webhook_url = "https://example.com/webhook"
    
    response = client.post(
        f"/api/v2/service/client/register/{client_id}",
        json={"webhook_url": webhook_url},
        headers=auth_headers,
    )
    
    assert response.status_code == 200
    data = response.json()
    assert data["client_id"] == client_id
    assert data["webhook_url"] == webhook_url
    assert "registered_at" in data


def test_client_registration_without_webhook(client, auth_headers):
    """Test client registration without webhook URL."""
    client_id = "test-client-2"
    
    response = client.post(
        f"/api/v2/service/client/register/{client_id}",
        headers=auth_headers,
    )
    
    assert response.status_code == 200
    data = response.json()
    assert data["client_id"] == client_id
    assert data["webhook_url"] == ""


def test_client_registration_requires_auth(client):
    """Test that client registration requires authentication."""
    client_id = "test-client-3"
    
    response = client.post(
        f"/api/v2/service/client/register/{client_id}",
        json={"webhook_url": "https://example.com/webhook"},
    )
    
    assert response.status_code == 401


def test_client_deregistration(client, auth_headers, db_session, sample_network):
    """Test client deregistration."""
    client_id = "test-client-4"
    
    # First register the client
    sample_network.connected_clients = {
        client_id: {
            "ip": "127.0.0.1",
            "connection_start_time": datetime.now(timezone.utc).isoformat(),
            "webhook_url": "https://example.com/webhook",
        }
    }
    db_session.commit()
    
    # Then deregister
    response = client.delete(
        f"/api/v2/service/client/deregister/{client_id}",
        headers=auth_headers,
    )
    
    assert response.status_code == 200
    data = response.json()
    assert data["client_id"] == client_id


def test_client_deregistration_not_found(client, auth_headers, sample_network):
    """Test deregistering non-existent client."""
    client_id = "non-existent-client"
    
    response = client.delete(
        f"/api/v2/service/client/deregister/{client_id}",
        headers=auth_headers,
    )
    
    assert response.status_code == 404


def test_service_enable(client, auth_headers):
    """Test enabling service for user's network."""
    response = client.patch(
        "/api/v2/service/enable",
        headers=auth_headers,
    )
    
    assert response.status_code == 200
    data = response.json()
    assert data["message"] == "Service enabled successfully"


def test_service_disable(client, auth_headers, sample_network):
    """Test disabling service for user's network."""
    response = client.patch(
        "/api/v2/service/disable",
        headers=auth_headers,
    )
    
    assert response.status_code == 200
    data = response.json()
    assert data["message"] == "Service disabled successfully"


@pytest.mark.asyncio
async def test_webhook_payload_structure():
    """Test webhook payload has correct structure."""
    interaction_id = str(uuid.uuid4())
    network_id = str(uuid.uuid4())
    speaker_id = str(uuid.uuid4())
    conversation_id = str(uuid.uuid4())
    
    payload = WebhookPayload(
        interaction_id=interaction_id,
        network_id=network_id,
        text="Test interaction",
        timestamp=datetime.now(timezone.utc),
        speaker_id=speaker_id,
        conversation_id=conversation_id,
    )
    
    # Verify all fields are present
    payload_dict = payload.model_dump()
    assert payload_dict["interaction_id"] == interaction_id
    assert payload_dict["network_id"] == network_id
    assert payload_dict["text"] == "Test interaction"
    assert payload_dict["speaker_id"] == speaker_id
    assert payload_dict["conversation_id"] == conversation_id
    assert "timestamp" in payload_dict


@pytest.mark.asyncio
async def test_webhook_dispatcher_sends_to_all_clients(db_session, sample_network):
    """Test that webhook dispatcher sends to all registered clients."""
    from app.services.webhook_dispatcher import WebhookDispatcher
    
    # Setup network with multiple clients
    sample_network.connected_clients = {
        "client1": {
            "ip": "127.0.0.1",
            "webhook_url": "https://example.com/webhook1",
        },
        "client2": {
            "ip": "127.0.0.2",
            "webhook_url": "https://example.com/webhook2",
        },
    }
    db_session.commit()
    
    # Create a test interaction
    interaction = Interaction(
        id=uuid.uuid4(),
        network_id=sample_network.id,
        text="Test interaction",
        timestamp=datetime.now(timezone.utc),
    )
    
    # Mock the HTTP client
    dispatcher = WebhookDispatcher()
    
    with patch.object(dispatcher, '_send_webhook', new_callable=AsyncMock) as mock_send:
        mock_send.return_value = True
        
        result = await dispatcher.dispatch_interaction(interaction, sample_network)
        
        # Verify webhook was sent to both clients
        assert mock_send.call_count == 2
        assert len(result) == 2
        assert result.get("client1") is True
        assert result.get("client2") is True


@pytest.mark.asyncio
async def test_webhook_dispatcher_handles_failures(db_session, sample_network):
    """Test that webhook dispatcher handles failures gracefully."""
    from app.services.webhook_dispatcher import WebhookDispatcher
    
    # Setup network with one client
    sample_network.connected_clients = {
        "client1": {
            "ip": "127.0.0.1",
            "webhook_url": "https://example.com/webhook1",
        },
    }
    db_session.commit()
    
    # Create a test interaction
    interaction = Interaction(
        id=uuid.uuid4(),
        network_id=sample_network.id,
        text="Test interaction",
        timestamp=datetime.now(timezone.utc),
    )
    
    # Mock the HTTP client to fail
    dispatcher = WebhookDispatcher()
    
    with patch.object(dispatcher, '_send_webhook', new_callable=AsyncMock) as mock_send:
        mock_send.side_effect = Exception("Network error")
        
        result = await dispatcher.dispatch_interaction(interaction, sample_network)
        
        # Verify failure was handled
        assert len(result) == 1
        assert result.get("client1") is False
