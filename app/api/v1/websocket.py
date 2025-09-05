"""
WebSocket endpoints for real-time notifications.
"""

from fastapi import APIRouter, WebSocket, WebSocketDisconnect, Depends, HTTPException, status
from sqlalchemy.orm import Session
import secrets

from app.core.websocket import ws_manager
from app.api.deps import get_db_dependency
from app.models.network import MiraNetwork
from app.models.client import MiraClient

router = APIRouter()


@router.websocket("/ws/{network_id}/{client_id}")
async def websocket_endpoint(
    websocket: WebSocket,
    network_id: str,
    client_id: str,
    auth_token: str,
    db: Session = Depends(get_db_dependency)
):
    """WebSocket endpoint for real-time notifications."""
    try:
        # Verify network and client
        network = db.query(MiraNetwork).filter(MiraNetwork.network_id == network_id).first()
        if not network:
            await websocket.close(code=status.WS_1008_POLICY_VIOLATION)
            return

        client = (
            db.query(MiraClient)
            .filter(
                MiraClient.network_id == network.id,
                MiraClient.client_id == client_id
            )
            .first()
        )

        if not client or not client.verify_auth_token(auth_token):
            await websocket.close(code=status.WS_1008_POLICY_VIOLATION)
            return

        # Generate unique WebSocket ID
        websocket_id = secrets.token_urlsafe(16)

        # Update client status
        client.update_websocket(websocket_id)
        db.commit()

        # Accept connection
        await ws_manager.connect(websocket, network.id, client_id, websocket_id)

        try:
            while True:
                # Wait for messages (only used for pings)
                data = await websocket.receive_text()
                if data == "ping":
                    await websocket.send_text("pong")
        except WebSocketDisconnect:
            pass
        finally:
            # Update client status
            client.update_websocket(None)
            db.commit()
            await ws_manager.disconnect(websocket_id)

    except Exception as e:
        await websocket.close(code=status.WS_1011_INTERNAL_ERROR)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"WebSocket error: {str(e)}"
        )
