"""
Service Registry for Multi-Network Support

This module provides a proper service registry pattern for managing services
across multiple networks with proper lifecycle management and cleanup.
"""

import threading
from datetime import datetime, timezone
from typing import Callable, Dict, Any
from dataclasses import dataclass
from app.core.mira_logger import MiraLogger


@dataclass
class ServiceConfig:
    """Configuration for a service within a network"""

    network_id: str
    created_at: datetime
    last_accessed: datetime
    ttl_seconds: int = 3600  # 1 hour default TTL


class ServiceRegistry:
    """
    Service registry for managing network-specific service instances.

    This follows the industry standard pattern of:
    1. Dependency injection
    2. Proper lifecycle management
    3. Resource cleanup
    4. Thread safety
    """

    def __init__(self, max_networks: int = 100, default_ttl_seconds: int = 3600):
        """
        Initialize the service registry.

        Args:
            max_networks: Maximum number of networks to keep in memory
            default_ttl_seconds: Default TTL for network services
        """
        self._services: Dict[str, Dict[str, Any]] = {}
        self._configs: Dict[str, ServiceConfig] = {}
        self._lock = threading.RLock()
        self._max_networks = max_networks
        self._default_ttl_seconds = default_ttl_seconds

        MiraLogger.info(f"ServiceRegistry initialized with max_networks={max_networks}")

    def get_service(self, network_id: str, service_name: str, service_factory: Callable) -> Any:
        """
        Get or create a service for a specific network.

        Args:
            network_id: ID of the network
            service_name: Name of the service to get
            service_factory: Factory function to create the service if it doesn't exist

        Returns:
            The requested service instance
        """
        with self._lock:
            self._cleanup_expired()

            # Initialize network if it doesn't exist
            if network_id not in self._services:
                if len(self._services) >= self._max_networks:
                    self._evict_oldest()
                self._initialize_network(network_id)

            # Get or create the specific service
            if service_name not in self._services[network_id]:
                self._services[network_id][service_name] = service_factory(network_id)
                MiraLogger.info(f"Created {service_name} for network {network_id}")

            # Update last accessed time
            self._configs[network_id].last_accessed = datetime.now(timezone.utc)

            return self._services[network_id][service_name]

    def _initialize_network(self, network_id: str):
        """Initialize a new network in the registry."""
        self._services[network_id] = {}
        self._configs[network_id] = ServiceConfig(
            network_id=network_id,
            created_at=datetime.now(timezone.utc),
            last_accessed=datetime.now(timezone.utc),
            ttl_seconds=self._default_ttl_seconds,
        )
        MiraLogger.info(f"Initialized network {network_id}")

    def _cleanup_expired(self):
        """Remove expired networks from the registry."""
        now = datetime.now(timezone.utc)
        expired_networks = []

        for network_id, config in self._configs.items():
            if (now - config.last_accessed).total_seconds() > config.ttl_seconds:
                expired_networks.append(network_id)

        for network_id in expired_networks:
            self._remove_network(network_id)
            MiraLogger.info(f"Removed expired network {network_id}")

    def _evict_oldest(self):
        """Remove the oldest network to make room for a new one."""
        if not self._configs:
            return

        oldest_network = min(self._configs.items(), key=lambda x: x[1].last_accessed)[0]

        self._remove_network(oldest_network)
        MiraLogger.info(f"Evicted oldest network {oldest_network}")

    def _remove_network(self, network_id: str):
        """Remove a network and all its services."""
        # Clean up any resources if needed
        if network_id in self._services:
            for service_name, service in self._services[network_id].items():
                if hasattr(service, "cleanup"):
                    try:
                        service.cleanup()
                    except Exception as e:
                        MiraLogger.warning(
                            f"Error cleaning up {service_name} for network {network_id}: {e}"
                        )

        self._services.pop(network_id, None)
        self._configs.pop(network_id, None)

    def remove_network(self, network_id: str):
        """Manually remove a network and all its services."""
        with self._lock:
            self._remove_network(network_id)
            MiraLogger.info(f"Manually removed network {network_id}")

    def get_network_stats(self) -> Dict[str, Any]:
        """Get statistics about the registry."""
        with self._lock:
            return {
                "total_networks": len(self._services),
                "max_networks": self._max_networks,
                "networks": {
                    network_id: {
                        "services": list(services.keys()),
                        "created_at": config.created_at.isoformat(),
                        "last_accessed": config.last_accessed.isoformat(),
                        "age_seconds": (
                            datetime.now(timezone.utc) - config.created_at
                        ).total_seconds(),
                    }
                    for network_id, services in self._services.items()
                    for config in [self._configs[network_id]]
                },
            }

    def cleanup_all(self):
        """Clean up all networks and services."""
        with self._lock:
            for network_id in list(self._services.keys()):
                self._remove_network(network_id)
            MiraLogger.info("Cleaned up all networks")


# Global service registry instance
service_registry = ServiceRegistry()
