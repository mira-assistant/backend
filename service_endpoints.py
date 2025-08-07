import logging
from datetime import datetime, timezone
from fastapi import APIRouter, HTTPException, Request

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/service", tags=["service"])


def get_status():
    from mira import status
    return status


def get_audio_scorer():
    from mira import audio_scorer
    return audio_scorer


def disable_service():
    status = get_status()
    status["enabled"] = False
    return {"message": "Service disabled successfully"}


@router.post("/client/register/{client_id}")
def register_client(client_id: str, request: Request):
    """Register a client and initialize stream scoring."""
    status = get_status()
    audio_scorer = get_audio_scorer()
    
    client_ip = request.client.host if request.client else "unknown"
    connection_start_time = datetime.now(timezone.utc)

    status["connected_clients"][client_id] = {
        "ip": client_ip,
        "connection_start_time": connection_start_time,
        "connection_runtime": 0.0,
    }

    success = audio_scorer.register_client(client_id=client_id)

    if success:
        logger.info(f"Client {client_id} registered for stream scoring from IP {client_ip}")

    return {"message": f"{client_id} registered successfully", "stream_scoring_enabled": success}


@router.delete("/client/deregister/{client_id}")
def deregister_client(client_id: str):
    """Deregister a client and remove from stream scoring."""
    status = get_status()
    audio_scorer = get_audio_scorer()
    
    if client_id in status["connected_clients"]:
        del status["connected_clients"][client_id]
    else:
        print("Client already deregistered or not found:", client_id)

    success = audio_scorer.deregister_client(client_id)

    if len(status["connected_clients"]) == 0:
        disable_service()
        logging.info("All clients deregistered, service disabled")

    if client_id not in status["connected_clients"] and not success:
        return {"message": f"{client_id} already deregistered or not found"}

    return {"message": f"{client_id} deregistered successfully", "stream_scoring_removed": success}


@router.patch("/enable")
def enable_service():
    status = get_status()
    status["enabled"] = True
    return {"message": "Service enabled successfully"}


@router.patch("/disable")
def disable_service_endpoint():
    return disable_service()


@router.get("/{client_id}")
def get_client_info(client_id: str):
    """Get detailed information about a specific client."""
    status = get_status()
    
    client_dict = status["connected_clients"].get(client_id)

    if not client_dict:
        raise HTTPException(status_code=404, detail=f"Client {client_id} not found")

    return client_dict