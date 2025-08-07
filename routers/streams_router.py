from mira import audio_scorer, logger
from fastapi import APIRouter, HTTPException, Body

router = APIRouter(prefix="/streams")


@router.get("/best")
def get_best_stream():
    """Get the currently selected best audio stream."""
    best_stream_info = audio_scorer.get_best_stream()

    if not best_stream_info:
        return {"best_stream": None}

    return {"best_stream": best_stream_info}


@router.get("/scores")
def get_all_stream_scores():
    """Get quality scores for all active streams."""

    try:
        clients_info = audio_scorer.clients
        scores = audio_scorer.get_all_stream_scores()
        best_stream = audio_scorer.get_best_stream()

        return {
            "active_streams": len(clients_info),
            "stream_scores": scores,
            "current_best": best_stream,
        }

    except Exception as e:
        logger.error(f"Error getting stream scores: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get stream scores: {str(e)}")


@router.get("/{client_id}/info")
def get_client_stream_info(client_id: str):
    """Get detailed stream information about a specific client."""

    if client_id not in audio_scorer.clients:
        raise HTTPException(
            status_code=404, detail=f"Client {client_id} not found in stream scoring"
        )

    client_info = audio_scorer.clients[client_id]
    current_score = audio_scorer.get_all_stream_scores().get(client_id, 0.0)
    best_stream = audio_scorer.get_best_stream()
    is_best_stream = best_stream and best_stream.get("client_id") == client_id

    return {
        "client_id": client_id,
        "quality_metrics": client_info.quality_metrics.__dict__,
        "current_score": round(current_score, 2),
        "is_best_stream": is_best_stream,
        "last_update": client_info.last_update,
    }


@router.post("/phone/location")
def update_phone_location(request: dict = Body(...)):
    """Update GPS-based location data for phone tracking."""
    try:
        client_id = request.get("client_id")
        location = request.get("location")

        if not client_id:
            raise HTTPException(status_code=400, detail="client_id is required")
        if not location:
            raise HTTPException(status_code=400, detail="location data is required")

        required_fields = ["latitude", "longitude"]
        for field in required_fields:
            if field not in location:
                raise HTTPException(status_code=400, detail=f"location.{field} is required")

        success = audio_scorer.set_phone_location(client_id, location)

        if not success:
            raise HTTPException(status_code=404, detail=f"Client {client_id} not found")

        logger.info(f"Updated phone location for {client_id}: {location}")

        return {
            "message": f"Phone location updated successfully for {client_id}",
            "location": location,
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error updating phone location: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to update phone location: {str(e)}")


@router.post("/phone/rssi")
def update_phone_rssi(request: dict = Body(...)):
    """Update RSSI-based proximity data for phone tracking to specific client."""
    try:
        phone_client_id = request.get("phone_client_id")
        target_client_id = request.get("target_client_id")
        rssi = request.get("rssi")

        if not phone_client_id:
            raise HTTPException(status_code=400, detail="phone_client_id is required")
        if not target_client_id:
            raise HTTPException(status_code=400, detail="target_client_id is required")
        if rssi is None:
            raise HTTPException(status_code=400, detail="rssi value is required")

        success = audio_scorer.set_phone_rssi(target_client_id, rssi)

        if not success:
            raise HTTPException(
                status_code=404, detail=f"Target client {target_client_id} not found"
            )

        logger.info(f"Updated RSSI from {phone_client_id} to {target_client_id}: {rssi} dBm")

        return {
            "message": f"RSSI updated successfully from {phone_client_id} to {target_client_id}",
            "rssi": rssi,
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error updating phone RSSI: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to update phone RSSI: {str(e)}")
