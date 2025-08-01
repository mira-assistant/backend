"""
Tests for audio recording stop functionality.
"""
from fastapi.testclient import TestClient


def test_audio_stop_functionality():
    """Test audio stop endpoint without loading ML models."""
    # Import here to avoid ML model loading issues
    import sys
    import os
    
    # Mock the ML model dependencies to avoid network issues
    import unittest.mock as mock
    
    with mock.patch('sentence_processor.get_models'), \
         mock.patch('context_processor.create_context_processor'):
        
        from mira import app
        client = TestClient(app)
        
        # Test audio stop endpoint
        response = client.post("/audio/stop")
        assert response.status_code == 200
        data = response.json()
        assert "message" in data
        assert "Audio recording stopped successfully" in data["message"]
        assert data["audio_recording"] is False
        assert data["vad_active"] is False


def test_audio_start_functionality():
    """Test audio start endpoint."""
    import unittest.mock as mock
    
    with mock.patch('sentence_processor.get_models'), \
         mock.patch('context_processor.create_context_processor'):
        
        from mira import app
        client = TestClient(app)
        
        # First enable the service
        enable_response = client.patch("/service/enable")
        assert enable_response.status_code == 200
        
        # Test audio start endpoint
        response = client.post("/audio/start")
        assert response.status_code == 200
        data = response.json()
        assert "message" in data
        assert "Audio recording started successfully" in data["message"]
        assert data["audio_recording"] is True
        assert data["vad_active"] is True


def test_audio_start_when_service_disabled():
    """Test audio start endpoint when service is disabled."""
    import unittest.mock as mock
    
    with mock.patch('sentence_processor.get_models'), \
         mock.patch('context_processor.create_context_processor'):
        
        from mira import app
        client = TestClient(app)
        
        # Ensure service is disabled
        disable_response = client.patch("/service/disable")
        assert disable_response.status_code == 200
        
        # Test audio start endpoint - should fail
        response = client.post("/audio/start")
        assert response.status_code == 200
        data = response.json()
        assert "Cannot start audio recording - service is disabled" in data["message"]
        assert data["audio_recording"] is False
        assert data["vad_active"] is False
        assert data["service_enabled"] is False


def test_audio_status_endpoint():
    """Test audio status endpoint."""
    import unittest.mock as mock
    
    with mock.patch('sentence_processor.get_models'), \
         mock.patch('context_processor.create_context_processor'):
        
        from mira import app
        client = TestClient(app)
        
        # Test audio status endpoint
        response = client.get("/audio/status")
        assert response.status_code == 200
        data = response.json()
        assert "service_enabled" in data
        assert "audio_recording" in data
        assert "vad_active" in data
        assert "listening_clients" in data
        assert "client_list" in data


def test_service_disable_stops_audio():
    """Test that service disable also stops audio recording."""
    import unittest.mock as mock
    
    with mock.patch('sentence_processor.get_models'), \
         mock.patch('context_processor.create_context_processor'):
        
        from mira import app
        client = TestClient(app)
        
        # Enable service and start audio
        client.patch("/service/enable")
        client.post("/audio/start")
        
        # Verify audio is started
        status_response = client.get("/audio/status")
        status_data = status_response.json()
        assert status_data["audio_recording"] is True
        assert status_data["vad_active"] is True
        
        # Disable service
        disable_response = client.patch("/service/disable")
        assert disable_response.status_code == 200
        
        # Verify audio is stopped
        status_response = client.get("/audio/status")
        status_data = status_response.json()
        assert status_data["service_enabled"] is False
        assert status_data["audio_recording"] is False
        assert status_data["vad_active"] is False


def test_interaction_register_respects_audio_state():
    """Test that interaction registration respects audio recording state."""
    import unittest.mock as mock
    
    with mock.patch('sentence_processor.get_models'), \
         mock.patch('context_processor.create_context_processor'):
        
        from mira import app
        client = TestClient(app)
        
        # Stop audio recording
        client.post("/audio/stop")
        
        # Try to register interaction - should be rejected
        response = client.post("/interactions/register", content=b"test audio data")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "rejected"
        assert "Audio recording is stopped" in data["message"]


if __name__ == "__main__":
    # Run a simple test to verify functionality
    test_audio_stop_functionality()
    test_audio_status_endpoint()
    print("Basic audio stop tests passed!")