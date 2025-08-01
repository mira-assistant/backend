"""
API tests for wake word detection endpoints.
"""

import pytest
from unittest.mock import Mock, patch
from fastapi.testclient import TestClient
import sys

# Mock heavy dependencies that aren't needed for API testing
sys.modules["db"] = Mock()
sys.modules["models"] = Mock()
sys.modules["inference_processor"] = Mock()
sys.modules["sentence_processor"] = Mock()
sys.modules["context_processor"] = Mock()

# Mock the database and processor modules
mock_db = Mock()
mock_db.query.return_value.filter_by.return_value.first.return_value = None
mock_db.query.return_value.order_by.return_value.limit.return_value.all.return_value = []


def mock_get_db_session():
    return mock_db


def mock_create_context_processor():
    return Mock()


# Apply mocks
with patch.dict(
    "sys.modules",
    {
        "db": Mock(get_db_session=mock_get_db_session),
        "models": Mock(Interaction=Mock, Person=Mock, Conversation=Mock),
        "context_processor": Mock(create_context_processor=mock_create_context_processor),
        "inference_processor": Mock(),
        "sentence_processor": Mock(),
    },
):
    from mira import app

client = TestClient(app)


class TestWakeWordAPI:
    """Test suite for wake word detection API endpoints"""

    def test_get_wake_words(self):
        """Test getting all wake words"""
        response = client.get("/wake-words")
        assert response.status_code == 200
        data = response.json()
        assert "wake_words" in data
        assert "stats" in data
        assert isinstance(data["wake_words"], dict)
        assert isinstance(data["stats"], dict)

        # Should have default wake words
        assert len(data["wake_words"]) > 0
        assert "hey mira" in data["wake_words"]

    def test_add_wake_word_basic(self):
        """Test adding a basic wake word"""
        wake_word_data = {"word": "hello assistant", "sensitivity": 0.8}
        response = client.post("/wake-words", json=wake_word_data)
        assert response.status_code == 200
        data = response.json()
        assert "message" in data
        assert "hello assistant" in data["message"]
        assert data["word"] == "hello assistant"
        assert data["sensitivity"] == 0.8

    def test_add_wake_word_full_config(self):
        """Test adding a wake word with full configuration"""
        wake_word_data = {
            "word": "custom trigger",
            "sensitivity": 0.9,
            "min_confidence": 0.7,
            "cooldown_seconds": 3.0,
        }
        response = client.post("/wake-words", json=wake_word_data)
        assert response.status_code == 200
        data = response.json()
        assert data["word"] == "custom trigger"
        assert data["sensitivity"] == 0.9
        assert data["min_confidence"] == 0.7
        assert data["cooldown_seconds"] == 3.0

    def test_add_empty_wake_word(self):
        """Test adding an empty wake word"""
        wake_word_data = {"word": ""}
        response = client.post("/wake-words", json=wake_word_data)
        assert response.status_code == 400
        assert "cannot be empty" in response.json()["detail"]

    def test_remove_wake_word_existing(self):
        """Test removing an existing wake word"""
        # First add a wake word
        client.post("/wake-words", json={"word": "test removal"})

        # Then remove it
        response = client.delete("/wake-words/test removal")
        assert response.status_code == 200
        data = response.json()
        assert "removed successfully" in data["message"]
        assert data["word"] == "test removal"

    def test_remove_wake_word_nonexistent(self):
        """Test removing a non-existent wake word"""
        response = client.delete("/wake-words/nonexistent word")
        assert response.status_code == 404
        assert "not found" in response.json()["detail"]

    def test_enable_wake_word(self):
        """Test enabling a wake word"""
        # First add a wake word
        client.post("/wake-words", json={"word": "test enable"})

        # Enable it
        response = client.patch("/wake-words/test enable/enable")
        assert response.status_code == 200
        data = response.json()
        assert "enabled" in data["message"]
        assert data["enabled"] is True
        assert data["word"] == "test enable"

    def test_disable_wake_word(self):
        """Test disabling a wake word"""
        # First add a wake word
        client.post("/wake-words", json={"word": "test disable"})

        # Disable it
        response = client.patch("/wake-words/test disable/disable")
        assert response.status_code == 200
        data = response.json()
        assert "disabled" in data["message"]
        assert data["enabled"] is False
        assert data["word"] == "test disable"

    def test_enable_nonexistent_wake_word(self):
        """Test enabling a non-existent wake word"""
        response = client.patch("/wake-words/nonexistent/enable")
        assert response.status_code == 404
        assert "not found" in response.json()["detail"]

    def test_disable_nonexistent_wake_word(self):
        """Test disabling a non-existent wake word"""
        response = client.patch("/wake-words/nonexistent/disable")
        assert response.status_code == 404
        assert "not found" in response.json()["detail"]

    def test_get_recent_detections(self):
        """Test getting recent detections"""
        response = client.get("/wake-words/detections")
        assert response.status_code == 200
        data = response.json()
        assert "detections" in data
        assert "count" in data
        assert "limit" in data
        assert isinstance(data["detections"], list)
        assert data["limit"] == 10  # Default limit

    def test_get_recent_detections_custom_limit(self):
        """Test getting recent detections with custom limit"""
        response = client.get("/wake-words/detections?limit=5")
        assert response.status_code == 200
        data = response.json()
        assert data["limit"] == 5

    def test_clear_detections(self):
        """Test clearing all detections"""
        response = client.delete("/wake-words/detections")
        assert response.status_code == 200
        data = response.json()
        assert "cleared" in data
        assert data["cleared"] is True

    def test_enable_wake_word_detection_system(self):
        """Test enabling the entire wake word detection system"""
        response = client.patch("/wake-words/enable")
        assert response.status_code == 200
        data = response.json()
        assert "enabled" in data["message"]
        assert data["enabled"] is True

    def test_disable_wake_word_detection_system(self):
        """Test disabling the entire wake word detection system"""
        response = client.patch("/wake-words/disable")
        assert response.status_code == 200
        data = response.json()
        assert "disabled" in data["message"]
        assert data["enabled"] is False

    def test_process_text_with_wake_word(self):
        """Test processing text that contains a wake word"""
        text_data = {
            "client_id": "test_client",
            "text": "Hey Mira, how are you today?",
            "audio_length": 2.5,
        }
        response = client.post("/wake-words/process", json=text_data)
        assert response.status_code == 200
        data = response.json()
        assert "detected" in data

        if data["detected"]:
            assert "wake_word" in data
            assert "confidence" in data
            assert "client_id" in data
            assert "timestamp" in data
            assert data["client_id"] == "test_client"

    def test_process_text_without_wake_word(self):
        """Test processing text that doesn't contain a wake word"""
        text_data = {
            "client_id": "test_client",
            "text": "Hello there, nice weather today",
            "audio_length": 1.5,
        }
        response = client.post("/wake-words/process", json=text_data)
        assert response.status_code == 200
        data = response.json()
        assert data["detected"] is False
        assert "message" in data

    def test_process_empty_text(self):
        """Test processing empty text"""
        text_data = {"client_id": "test_client", "text": ""}
        response = client.post("/wake-words/process", json=text_data)
        assert response.status_code == 400
        assert "cannot be empty" in response.json()["detail"]

    def test_wake_word_workflow(self):
        """Test complete wake word workflow"""
        # 1. Get initial wake words
        response = client.get("/wake-words")
        assert response.status_code == 200
        initial_count = len(response.json()["wake_words"])

        # 2. Add a custom wake word
        custom_word = "workflow test word"
        add_response = client.post(
            "/wake-words", json={"word": custom_word, "sensitivity": 0.8, "min_confidence": 0.6}
        )
        assert add_response.status_code == 200

        # 3. Verify wake word was added
        response = client.get("/wake-words")
        assert response.status_code == 200
        wake_words = response.json()["wake_words"]
        assert len(wake_words) == initial_count + 1
        assert custom_word in wake_words

        # 4. Test detection with the new wake word
        process_response = client.post(
            "/wake-words/process",
            json={
                "client_id": "workflow_client",
                "text": f"Please {custom_word} and start listening",
            },
        )
        assert process_response.status_code == 200

        # 5. Check if detection occurred
        if process_response.json()["detected"]:
            assert process_response.json()["wake_word"] == custom_word

        # 6. Disable the wake word
        disable_response = client.patch(f"/wake-words/{custom_word}/disable")
        assert disable_response.status_code == 200

        # 7. Test detection is disabled
        process_response2 = client.post(
            "/wake-words/process",
            json={
                "client_id": "workflow_client",
                "text": f"Please {custom_word} and start listening",
            },
        )
        assert process_response2.status_code == 200
        # Should not detect the disabled wake word
        if process_response2.json()["detected"]:
            assert process_response2.json()["wake_word"] != custom_word

        # 8. Remove the wake word
        remove_response = client.delete(f"/wake-words/{custom_word}")
        assert remove_response.status_code == 200

        # 9. Verify wake word was removed
        final_response = client.get("/wake-words")
        assert response.status_code == 200
        final_wake_words = final_response.json()["wake_words"]
        assert custom_word not in final_wake_words


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
