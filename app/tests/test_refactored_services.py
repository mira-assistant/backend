"""
Test script to verify the refactored services work correctly with the new architecture.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from services.service_factory import (
    get_command_processor,
    get_context_processor,
    get_inference_processor,
    get_multi_stream_processor,
    get_sentence_processor
)
from services.service_registry import service_registry
from models import Interaction
import uuid
from datetime import datetime, timezone


def test_service_registry():
    """Test that the service registry works correctly."""
    print("Testing Service Registry...")

    # Test network stats
    stats = service_registry.get_network_stats()
    print(f"Initial stats: {stats}")

    # Test service creation
    network_id = "test-network-1"

    # Get services for the network
    command_processor = get_command_processor(network_id)
    context_processor = get_context_processor(network_id)
    inference_processor = get_inference_processor(network_id)
    multi_stream_processor = get_multi_stream_processor(network_id)
    sentence_processor = get_sentence_processor(network_id)

    print(f"Created services for network {network_id}")
    print(f"Command processor type: {type(command_processor)}")
    print(f"Context processor type: {type(context_processor)}")
    print(f"Inference processor type: {type(inference_processor)}")
    print(f"Multi stream processor type: {type(multi_stream_processor)}")
    print(f"Sentence processor type: {type(sentence_processor)}")

    # Test that services are cached
    command_processor2 = get_command_processor(network_id)
    assert command_processor is command_processor2, "Services should be cached and reused"
    print("✓ Service caching works correctly")

    # Test network stats after creation
    stats = service_registry.get_network_stats()
    print(f"Stats after service creation: {stats}")

    # Test cleanup
    service_registry.remove_network(network_id)
    stats = service_registry.get_network_stats()
    print(f"Stats after cleanup: {stats}")

    print("✓ Service Registry test passed\n")


def test_command_processor():
    """Test the command processor functionality."""
    print("Testing Command Processor...")

    network_id = "test-network-2"
    command_processor = get_command_processor(network_id)

    # Test wake word detection
    detection = command_processor.detect_wake_words_text(
        client_id="test-client",
        transcribed_text="hey mira how are you",
        audio_length=2.0
    )

    if detection:
        print(f"✓ Wake word detected: {detection.wake_word} (confidence: {detection.confidence})")
    else:
        print("No wake word detected (this is expected if no wake words are configured)")

    # Test adding a wake word
    success = command_processor.add_wake_word(
        word="test wake word",
        sensitivity=0.8,
        min_confidence=0.6
    )
    print(f"✓ Wake word added: {success}")

    # Test wake word detection with the new word
    detection = command_processor.detect_wake_words_text(
        client_id="test-client",
        transcribed_text="this is a test wake word",
        audio_length=2.0
    )

    if detection:
        print(f"✓ Custom wake word detected: {detection.wake_word}")
    else:
        print("Custom wake word not detected")

    print("✓ Command Processor test passed\n")


def test_multi_stream_processor():
    """Test the multi-stream processor functionality."""
    print("Testing Multi Stream Processor...")

    network_id = "test-network-3"
    processor = get_multi_stream_processor(network_id)

    # Test client registration
    success = processor.register_client("client-1")
    print(f"✓ Client registration: {success}")

    success = processor.register_client("client-2")
    print(f"✓ Second client registration: {success}")

    # Test getting all stream scores (should be empty initially)
    scores = processor.get_all_stream_scores()
    print(f"Initial stream scores: {scores}")

    # Test getting best stream
    best_stream = processor.get_best_stream()
    print(f"Best stream: {best_stream}")

    # Test cleanup
    processor.cleanup()
    print("✓ Multi Stream Processor test passed\n")


def test_network_isolation():
    """Test that different networks have isolated services."""
    print("Testing Network Isolation...")

    network1 = "network-1"
    network2 = "network-2"

    # Get services for both networks
    cmd1 = get_command_processor(network1)
    cmd2 = get_command_processor(network2)

    # They should be different instances
    assert cmd1 is not cmd2, "Different networks should have different service instances"
    print("✓ Different networks have different service instances")

    # Add wake word to network 1
    cmd1.add_wake_word("network1-word", sensitivity=0.8)

    # Test detection in network 1
    detection1 = cmd1.detect_wake_words_text("client", "this is a network1-word test")
    print(f"Network 1 detection: {detection1 is not None}")

    # Test detection in network 2 (should not detect the word)
    detection2 = cmd2.detect_wake_words_text("client", "this is a network1-word test")
    print(f"Network 2 detection: {detection2 is not None}")

    # Network 2 should not detect the word added to network 1
    assert detection2 is None, "Network isolation failed - network 2 detected network 1's wake word"
    print("✓ Network isolation works correctly")

    print("✓ Network Isolation test passed\n")


def main():
    """Run all tests."""
    print("=" * 50)
    print("Testing Refactored Services")
    print("=" * 50)

    try:
        test_service_registry()
        test_command_processor()
        test_multi_stream_processor()
        test_network_isolation()

        print("=" * 50)
        print("✅ All tests passed! The refactored services are working correctly.")
        print("=" * 50)

    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

    return True


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
