from mira import (
    status,
    audio_scorer,
    logger,
)
from fastapi import HTTPException, Request, APIRouter
from datetime import datetime, timezone

router = APIRouter(prefix="/service")


@router.post("/client/register/{client_id}")
def register_client(client_id: str, request: Request):
    """Register a client and initialize stream scoring."""

    client_ip = request.client.host if request.client else "unknown"
    connection_start_time = datetime.now(timezone.utc)

    status["connected_clients"][client_id] = {
        "ip": client_ip,
        "connection_start_time": connection_start_time,
        "connection_uptime": 0.0,
    }

    success = audio_scorer.register_client(client_id=client_id)

    if success:
        logger.info(f"Client {client_id} registered for stream scoring from IP {client_ip}")

    return {"message": f"{client_id} registered successfully", "stream_scoring_enabled": success}


@router.delete("/client/deregister/{client_id}")
def deregister_client(client_id: str):
    """Deregister a client and remove from stream scoring."""
    if client_id in status["connected_clients"]:
        del status["connected_clients"][client_id]
    else:
        print("Client already deregistered or not found:", client_id)

    success = audio_scorer.deregister_client(client_id)

    if len(status["connected_clients"]) == 0:
        disable_service()
        logger.info("All clients deregistered, service disabled")

    if client_id not in status["connected_clients"] and not success:
        return {"message": f"{client_id} already deregistered or not found"}

    return {"message": f"{client_id} deregistered successfully", "stream_scoring_removed": success}


@router.get("/{client_id}")
def get_client_info(client_id: str):
    """Get detailed information about a specific client."""

    client_dict = status["connected_clients"].get(client_id)

    if not client_dict:
        raise HTTPException(status_code=404, detail=f"Client {client_id} not found")

    return client_dict


@router.patch("/enable")
def enable_service():
    status["enabled"] = True
    return {"message": "Service enabled successfully"}


@router.patch("/disable")
def disable_service():
    status["enabled"] = False
    return {"message": "Service disabled successfully"}
