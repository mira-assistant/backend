"""
Simple test for the service registry without ML dependencies.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from services.service_registry import ServiceRegistry


def test_service_registry_basic():
    """Test basic service registry functionality."""
    print("Testing Service Registry...")

    # Create a test registry
    registry = ServiceRegistry(max_networks=5, default_ttl_seconds=60)

    # Test network stats
    stats = registry.get_network_stats()
    print(f"Initial stats: {stats}")
    assert stats['total_networks'] == 0
    print("✓ Initial stats correct")

    # Test service creation with a simple factory
    def create_test_service(network_id):
        return f"test-service-for-{network_id}"

    network_id = "test-network-1"
    service = registry.get_service(network_id, 'test_service', create_test_service)

    assert service == f"test-service-for-{network_id}"
    print(f"✓ Service created: {service}")

    # Test that services are cached
    service2 = registry.get_service(network_id, 'test_service', create_test_service)
    assert service is service2
    print("✓ Service caching works")

    # Test network stats after creation
    stats = registry.get_network_stats()
    print(f"Stats after creation: {stats}")
    assert stats['total_networks'] == 1
    print("✓ Stats updated correctly")

    # Test cleanup
    registry.remove_network(network_id)
    stats = registry.get_network_stats()
    assert stats['total_networks'] == 0
    print("✓ Cleanup works correctly")

    print("✅ Service Registry test passed!")


def test_network_isolation():
    """Test that different networks are isolated."""
    print("\nTesting Network Isolation...")

    registry = ServiceRegistry()

    def create_networked_service(network_id):
        return {"network_id": network_id, "data": f"data-for-{network_id}"}

    # Create services for different networks
    service1 = registry.get_service("network-1", 'service', create_networked_service)
    service2 = registry.get_service("network-2", 'service', create_networked_service)

    # They should be different instances
    assert service1 is not service2
    assert service1['network_id'] == "network-1"
    assert service2['network_id'] == "network-2"
    print("✓ Different networks have different service instances")

    # Test that services are properly isolated
    service1['data'] = "modified-data"
    assert service2['data'] == "data-for-network-2"  # Should not be affected
    print("✓ Network isolation works correctly")

    print("✅ Network Isolation test passed!")


def test_cleanup_and_eviction():
    """Test cleanup and eviction functionality."""
    print("\nTesting Cleanup and Eviction...")

    registry = ServiceRegistry(max_networks=2, default_ttl_seconds=1)

    def create_service(network_id):
        return f"service-{network_id}"

    # Create services up to the limit
    service1 = registry.get_service("network-1", 'service', create_service)
    service2 = registry.get_service("network-2", 'service', create_service)

    stats = registry.get_network_stats()
    assert stats['total_networks'] == 2
    print("✓ Created services up to limit")

    # Try to create another service (should evict oldest)
    service3 = registry.get_service("network-3", 'service', create_service)

    stats = registry.get_network_stats()
    assert stats['total_networks'] == 2  # Should still be 2 due to eviction
    print("✓ Eviction works correctly")

    # Test cleanup all
    registry.cleanup_all()
    stats = registry.get_network_stats()
    assert stats['total_networks'] == 0
    print("✓ Cleanup all works correctly")

    print("✅ Cleanup and Eviction test passed!")


def main():
    """Run all tests."""
    print("=" * 50)
    print("Testing Service Registry Architecture")
    print("=" * 50)

    try:
        test_service_registry_basic()
        test_network_isolation()
        test_cleanup_and_eviction()

        print("\n" + "=" * 50)
        print("✅ All tests passed! The service registry architecture is working correctly.")
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
