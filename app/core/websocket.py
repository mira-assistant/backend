"""
WebSocket manager for real-time notifications.
"""

from typing import Dict, Set, Optional
from fastapi import WebSocket, WebSocketDisconnect
import json
from datetime import datetime
import asyncio
import logging
from uuid import UUID

logger = logging.getLogger(__name__)


class WebSocketManager:
    """Manages WebSocket connections and notifications."""

    def __init__(self):
        # network_id -> {client_id -> websocket}
        self._connections: Dict[UUID, Dict[str, WebSocket]] = {}
        # websocket_id -> (network_id, client_id)
        self._socket_lookup: Dict[str, tuple[UUID, str]] = {}
        # Track active pings
        self._ping_tasks: Dict[str, asyncio.Task] = {}

    async def connect(
        self,
        websocket: WebSocket,
        network_id: UUID,
        client_id: str,
        websocket_id: str
    ) -> None:
        """Connect a new WebSocket client."""
        await websocket.accept()

        if network_id not in self._connections:
            self._connections[network_id] = {}

        # Disconnect existing connection if any
        if client_id in self._connections[network_id]:
            try:
                await self._connections[network_id][client_id].close()
            except:
                pass

        self._connections[network_id][client_id] = websocket
        self._socket_lookup[websocket_id] = (network_id, client_id)

        # Start ping task
        self._ping_tasks[websocket_id] = asyncio.create_task(
            self._ping_client(websocket, websocket_id)
        )

    async def disconnect(self, websocket_id: str) -> None:
        """Disconnect a WebSocket client."""
        if websocket_id not in self._socket_lookup:
            return

        network_id, client_id = self._socket_lookup[websocket_id]

        # Stop ping task
        if websocket_id in self._ping_tasks:
            self._ping_tasks[websocket_id].cancel()
            del self._ping_tasks[websocket_id]

        # Remove connection
        if network_id in self._connections:
            if client_id in self._connections[network_id]:
                try:
                    await self._connections[network_id][client_id].close()
                except:
                    pass
                del self._connections[network_id][client_id]

            if not self._connections[network_id]:
                del self._connections[network_id]

        del self._socket_lookup[websocket_id]

    async def broadcast_to_network(
        self,
        network_id: UUID,
        message: dict,
        exclude_client: Optional[str] = None
    ) -> None:
        """Broadcast message to all clients in a network."""
        if network_id not in self._connections:
            return

        message_json = json.dumps({
            "timestamp": datetime.utcnow().isoformat(),
            "data": message
        })

        for client_id, websocket in self._connections[network_id].items():
            if client_id != exclude_client:
                try:
                    await websocket.send_text(message_json)
                except Exception as e:
                    logger.error(f"Failed to send to client {client_id}: {str(e)}")

    async def send_to_client(
        self,
        network_id: UUID,
        client_id: str,
        message: dict
    ) -> bool:
        """Send message to a specific client."""
        if (
            network_id not in self._connections
            or client_id not in self._connections[network_id]
        ):
            return False

        message_json = json.dumps({
            "timestamp": datetime.utcnow().isoformat(),
            "data": message
        })

        try:
            await self._connections[network_id][client_id].send_text(message_json)
            return True
        except Exception as e:
            logger.error(f"Failed to send to client {client_id}: {str(e)}")
            return False

    async def _ping_client(self, websocket: WebSocket, websocket_id: str) -> None:
        """Keep connection alive with periodic pings."""
        try:
            while True:
                try:
                    await websocket.send_text(json.dumps({"type": "ping"}))
                    await asyncio.sleep(30)  # Ping every 30 seconds
                except WebSocketDisconnect:
                    break
                except Exception as e:
                    logger.error(f"Ping failed for {websocket_id}: {str(e)}")
                    break
        finally:
            await self.disconnect(websocket_id)


# Global WebSocket manager instance
ws_manager = WebSocketManager()
