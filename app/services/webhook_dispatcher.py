"""
Webhook dispatcher service for sending interaction updates to registered clients.
"""

import asyncio
from typing import Dict, List, Optional
from datetime import datetime

import httpx
from sqlalchemy.orm import Session

import app.models as models
from app.core.mira_logger import MiraLogger
from app.schemas.client import WebhookPayload


class WebhookDispatcher:
    """Service for dispatching webhook notifications to registered clients."""

    def __init__(self):
        """Initialize webhook dispatcher."""
        self.logger = MiraLogger.get_logger(__name__)
        self._client: Optional[httpx.AsyncClient] = None

    async def _get_client(self) -> httpx.AsyncClient:
        """Get or create async HTTP client."""
        if self._client is None:
            self._client = httpx.AsyncClient(timeout=10.0)
        return self._client

    async def close(self):
        """Close HTTP client."""
        if self._client:
            await self._client.aclose()
            self._client = None

    async def dispatch_interaction(
        self,
        interaction: models.Interaction,
        network: models.MiraNetwork,
    ) -> Dict[str, bool]:
        """
        Dispatch interaction to all registered client webhooks.
        
        Args:
            interaction: The interaction to dispatch
            network: The network containing registered clients
            
        Returns:
            Dictionary mapping client_id to success status
        """
        connected_clients = network.connected_clients or {}
        
        if not connected_clients:
            self.logger.info(f"No clients registered for network {network.id}")
            return {}

        # Build webhook payload
        payload = WebhookPayload(
            interaction_id=str(interaction.id),
            network_id=str(interaction.network_id),
            text=interaction.text,
            timestamp=interaction.timestamp or datetime.utcnow(),
            speaker_id=str(interaction.speaker_id) if interaction.speaker_id else None,
            conversation_id=str(interaction.conversation_id) if interaction.conversation_id else None,
        )

        # Dispatch to all clients concurrently
        tasks = []
        client_ids = []
        
        for client_id, client_info in connected_clients.items():
            webhook_url = client_info.get("webhook_url")
            if webhook_url:
                tasks.append(self._send_webhook(client_id, webhook_url, payload))
                client_ids.append(client_id)

        if not tasks:
            self.logger.info(f"No webhook URLs configured for network {network.id}")
            return {}

        # Execute all webhook calls concurrently
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Build result map
        result_map = {}
        for client_id, result in zip(client_ids, results):
            if isinstance(result, Exception):
                self.logger.error(f"Webhook failed for {client_id}: {result}")
                result_map[client_id] = False
            else:
                result_map[client_id] = result

        return result_map

    async def _send_webhook(
        self,
        client_id: str,
        webhook_url: str,
        payload: WebhookPayload,
    ) -> bool:
        """
        Send webhook to a single client.
        
        Args:
            client_id: ID of the client
            webhook_url: URL to send webhook to
            payload: Webhook payload
            
        Returns:
            True if successful, False otherwise
        """
        try:
            client = await self._get_client()
            response = await client.post(
                webhook_url,
                json=payload.model_dump(mode="json"),
                headers={"Content-Type": "application/json"},
            )
            
            if response.status_code in [200, 201, 202, 204]:
                self.logger.info(
                    f"Webhook sent successfully to {client_id} at {webhook_url}"
                )
                return True
            else:
                self.logger.warning(
                    f"Webhook to {client_id} returned status {response.status_code}"
                )
                return False
                
        except httpx.TimeoutException:
            self.logger.error(f"Webhook timeout for {client_id} at {webhook_url}")
            return False
        except Exception as e:
            self.logger.error(
                f"Webhook error for {client_id} at {webhook_url}: {str(e)}"
            )
            return False


# Global webhook dispatcher instance
_webhook_dispatcher: Optional[WebhookDispatcher] = None


def get_webhook_dispatcher() -> WebhookDispatcher:
    """Get or create global webhook dispatcher instance."""
    global _webhook_dispatcher
    if _webhook_dispatcher is None:
        _webhook_dispatcher = WebhookDispatcher()
    return _webhook_dispatcher


async def cleanup_webhook_dispatcher():
    """Cleanup global webhook dispatcher."""
    global _webhook_dispatcher
    if _webhook_dispatcher:
        await _webhook_dispatcher.close()
        _webhook_dispatcher = None
