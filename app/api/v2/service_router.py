import uuid
from datetime import datetime, timezone

from fastapi import APIRouter, Depends, HTTPException, Path, Request
from sqlalchemy.orm import Session

import app.db as db
import app.models as models
import app.schemas.client as client_schemas
from app.api.deps import get_current_user_required

router = APIRouter(prefix="/service", tags=["service"])


@router.post("/client/register/{client_id}", response_model=client_schemas.ClientRegistrationResponse)
async def register_client(
    request: Request,
    client_id: str = Path(..., description="The ID of the client"),
    registration_data: client_schemas.ClientRegistrationRequest = None,
    current_user: models.User = Depends(get_current_user_required),
    db_session: Session = Depends(db.get_db),
):
    """
    Register a client with webhook URL for interaction updates.
    
    Requires JWT authentication. The network_id is extracted from the access token.
    """
    # Extract network_id from authenticated user
    network_id = str(current_user.id)
    
    client_ip = request.client.host if request.client else "unknown"
    connection_start_time = datetime.now(timezone.utc)

    # Find or create network for user
    network = (
        db_session.query(models.MiraNetwork)
        .filter(models.MiraNetwork.id == uuid.UUID(network_id))
        .first()
    )

    if not network:
        # Create network for user if it doesn't exist
        network = models.MiraNetwork(
            id=uuid.UUID(network_id),
            name=f"Network for {current_user.username or current_user.email}",
            service_enabled=True,
            connected_clients={},
        )
        db_session.add(network)

    # Prepare client info
    client_info = {
        "ip": client_ip,
        "connection_start_time": connection_start_time.isoformat(),
    }
    
    # Add webhook URL if provided
    if registration_data and registration_data.webhook_url:
        client_info["webhook_url"] = str(registration_data.webhook_url)

    # Update connected clients
    if not network.connected_clients:
        network.connected_clients = {}
    
    network.connected_clients[client_id] = client_info  # type: ignore

    db_session.commit()

    return client_schemas.ClientRegistrationResponse(
        message=f"{client_id} registered successfully",
        client_id=client_id,
        webhook_url=client_info.get("webhook_url", ""),
        registered_at=connection_start_time,
    )


@router.delete("/client/deregister/{client_id}", response_model=client_schemas.ClientDeregistrationResponse)
async def deregister_client(
    client_id: str = Path(..., description="The ID of the client"),
    current_user: models.User = Depends(get_current_user_required),
    db_session: Session = Depends(db.get_db),
):
    """
    Deregister a client and remove from stream scoring.
    
    Requires JWT authentication. The network_id is extracted from the access token.
    """
    # Extract network_id from authenticated user
    network_id = str(current_user.id)

    network = (
        db_session.query(models.MiraNetwork)
        .filter(models.MiraNetwork.id == uuid.UUID(network_id))
        .first()
    )

    if not network:
        raise HTTPException(status_code=404, detail=f"Network {network_id} not found")

    if not network.connected_clients or client_id not in network.connected_clients:
        raise HTTPException(status_code=404, detail=f"Client {client_id} not found")

    network.connected_clients.pop(client_id)
    db_session.commit()

    return client_schemas.ClientDeregistrationResponse(
        message=f"{client_id} deregistered successfully",
        client_id=client_id,
    )


@router.patch("/enable")
async def enable_service(
    current_user: models.User = Depends(get_current_user_required),
    db_session: Session = Depends(db.get_db),
):
    """
    Enable service for the authenticated user's network.
    
    Requires JWT authentication. The network_id is extracted from the access token.
    """
    # Extract network_id from authenticated user
    network_id = str(current_user.id)
    
    network = (
        db_session.query(models.MiraNetwork)
        .filter(models.MiraNetwork.id == uuid.UUID(network_id))
        .first()
    )

    if not network:
        # Create network for user if it doesn't exist
        network = models.MiraNetwork(
            id=uuid.UUID(network_id),
            name=f"Network for {current_user.username or current_user.email}",
            service_enabled=True,
            connected_clients={},
        )
        db_session.add(network)
    else:
        network.service_enabled = True  # type: ignore

    db_session.commit()

    return {"message": "Service enabled successfully"}


@router.patch("/disable")
async def disable_service(
    current_user: models.User = Depends(get_current_user_required),
    db_session: Session = Depends(db.get_db),
):
    """
    Disable service for the authenticated user's network.
    
    Requires JWT authentication. The network_id is extracted from the access token.
    """
    # Extract network_id from authenticated user
    network_id = str(current_user.id)
    
    network = (
        db_session.query(models.MiraNetwork)
        .filter(models.MiraNetwork.id == uuid.UUID(network_id))
        .first()
    )

    if not network:
        raise HTTPException(status_code=404, detail=f"Network {network_id} not found")

    network.service_enabled = False  # type: ignore
    db_session.commit()

    return {"message": "Service disabled successfully"}
