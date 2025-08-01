"""
Comprehensive tests for the audio recording stop functionality.
"""
from fastapi.testclient import TestClient
import unittest.mock as mock


def test_complete_audio_workflow():
    """Test the complete audio workflow including start, stop, and state management."""
    
    with mock.patch('sentence_processor.get_models'), \
         mock.patch('context_processor.create_context_processor'):
        
        from mira import app
        client = TestClient(app)
        
        print("Testing complete audio workflow...")
        
        # 1. Check initial status
        response = client.get("/audio/status")
        data = response.json()
        print(f"Initial status: {data}")
        assert data["service_enabled"] is False
        assert data["audio_recording"] is False
        assert data["vad_active"] is False
        
        # 2. Try to start audio without enabling service (should fail)
        response = client.post("/audio/start")
        data = response.json()
        print(f"Start audio without service enabled: {data}")
        assert "Cannot start audio recording - service is disabled" in data["message"]
        assert data["audio_recording"] is False
        
        # 3. Enable service
        response = client.patch("/service/enable")
        assert response.status_code == 200
        print("Service enabled")
        
        # 4. Start audio recording
        response = client.post("/audio/start")
        data = response.json()
        print(f"Start audio recording: {data}")
        assert "Audio recording started successfully" in data["message"]
        assert data["audio_recording"] is True
        assert data["vad_active"] is True
        
        # 5. Check status after starting audio
        response = client.get("/audio/status")
        data = response.json()
        print(f"Status after starting audio: {data}")
        assert data["service_enabled"] is True
        assert data["audio_recording"] is True
        assert data["vad_active"] is True
        
        # 6. Stop audio recording
        response = client.post("/audio/stop")
        data = response.json()
        print(f"Stop audio recording: {data}")
        assert "Audio recording stopped successfully" in data["message"]
        assert data["audio_recording"] is False
        assert data["vad_active"] is False
        assert data["previous_states"]["audio_recording"] is True
        assert data["previous_states"]["vad_active"] is True
        
        # 7. Check status after stopping audio
        response = client.get("/audio/status")
        data = response.json()
        print(f"Status after stopping audio: {data}")
        assert data["service_enabled"] is True  # Service still enabled
        assert data["audio_recording"] is False
        assert data["vad_active"] is False
        
        # 8. Test that interaction registration is rejected when audio is stopped
        response = client.post("/interactions/register", 
                              content=b"test audio data",
                              headers={"Content-Type": "application/octet-stream"})
        data = response.json()
        print(f"Interaction registration when audio stopped: {data}")
        assert data["status"] == "rejected"
        assert "Audio recording is stopped" in data["message"]
        
        # 9. Restart audio and test interaction registration works
        response = client.post("/audio/start")
        assert response.json()["audio_recording"] is True
        
        # This would normally work if we had proper audio data and models
        response = client.post("/interactions/register", 
                              content=b"test audio data",
                              headers={"Content-Type": "application/octet-stream"})
        # Since we don't have real audio data, it should still be rejected but for different reason
        print(f"Interaction registration when audio active: {response.json()}")
        
        # 10. Test service disable stops everything
        response = client.patch("/service/disable")
        assert response.status_code == 200
        
        response = client.get("/audio/status")
        data = response.json()
        print(f"Status after service disable: {data}")
        assert data["service_enabled"] is False
        assert data["audio_recording"] is False
        assert data["vad_active"] is False
        
        print("All tests passed successfully!")


if __name__ == "__main__":
    test_complete_audio_workflow()