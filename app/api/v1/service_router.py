from fastapi import Depends, HTTPException, Request, APIRouter, Path
from datetime import datetime, timezone

from sqlalchemy.orm import Session

import app.db as db
import app.models as models

router = APIRouter(prefix="/{network_id}/service")


@router.post("/client/register/{client_id}")
def register_client(
    request: Request,
    network_id: str = Path(..., description="The ID of the network"),
    client_id: str = Path(..., description="The ID of the client"),
    db: Session = Depends(db.get_db),
):
    """Register a client and initialize stream scoring."""

    client_ip = request.client.host if request.client else "unknown"
    connection_start_time = datetime.now(timezone.utc)

    network = db.query(models.MiraNetwork).filter(models.MiraNetwork.id == network_id).first()

    if not network:
        raise HTTPException(status_code=404, detail=f"Network {network_id} not found")

    network.connected_clients.append(
        {
            client_id: {
                "ip": client_ip,
                "connection_start_time": connection_start_time,
            }
        }
    )

    db.commit()

    return {"message": f"{client_id} registered successfully"}


@router.delete("/client/deregister/{client_id}")
def deregister_client(
    network_id: str = Path(..., description="The ID of the network"),
    client_id: str = Path(..., description="The ID of the client"),
    db: Session = Depends(db.get_db),
):
    """Deregister a client and remove from stream scoring."""

    network = db.query(models.MiraNetwork).filter(models.MiraNetwork.id == network_id).first()

    if not network:
        raise HTTPException(status_code=404, detail=f"Network {network_id} not found")

    network.connected_clients.pop(client_id)
    db.commit()

    return {"message": f"{client_id} deregistered successfully"}


@router.patch("/enable")
def enable_service(
    network_id: str = Path(..., description="The ID of the network"),
    db: Session = Depends(db.get_db),
):
    network = db.query(models.MiraNetwork).filter(models.MiraNetwork.id == network_id).first()

    if not network:
        raise HTTPException(status_code=404, detail=f"Network {network_id} not found")

    network.service_enabled = True
    db.commit()

    return {"message": "Service enabled successfully"}


@router.patch("/disable")
def disable_service(
    network_id: str = Path(..., description="The ID of the network"),
    db: Session = Depends(db.get_db),
):
    network = db.query(models.MiraNetwork).filter(models.MiraNetwork.id == network_id).first()

    if not network:
        raise HTTPException(status_code=404, detail=f"Network {network_id} not found")

    network.service_enabled = False
    db.commit()

    return {"message": "Service disabled successfully"}
