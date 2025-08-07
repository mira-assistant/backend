from mira import status, context_processor, audio_scorer, wake_word_detector, command_processor, inference_processor, logger
from fastapi import Request, APIRouter
import logging
from datetime import datetime, timezone

router = APIRouter(prefix="/service")

@router.post("/client/register/{client_id}")
def register_client(client_id: str, request: Request):
    """Register a client and initialize stream scoring."""
    # Get client IP address and connection start time
    client_ip = request.client.host if request.client else "unknown"
    connection_start_time = datetime.now(timezone.utc)

    # Store client information in connected_clients dictionary
    status["connected_clients"][client_id] = {
        "ip": client_ip,
        "connection_start_time": connection_start_time,
        "connection_runtime": 0.0,  # Runtime in seconds, updated dynamically
    }

    # Register client with audio stream scorer
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

    # Deregister from audio stream scorer
    success = audio_scorer.deregister_client(client_id)

    if len(status["connected_clients"]) == 0:
        disable_service()
        logging.info("All clients deregistered, service disabled")

    if client_id not in status["connected_clients"] and not success:
        return {"message": f"{client_id} already deregistered or not found"}

    return {"message": f"{client_id} deregistered successfully", "stream_scoring_removed": success}


@router.patch("/enable")
def enable_service():
    status["enabled"] = True
    return {"message": "Service enabled successfully"}


@router.patch("/disable")
def disable_service():
    status["enabled"] = False
    return {"message": "Service disabled successfully"}
