import uuid

import db as db
import models as models
from core.mira_logger import MiraLogger
from fastapi import APIRouter, Body, Depends, HTTPException, Path
from services.service_factory import get_multi_stream_processor
from sqlalchemy.orm import Session

router = APIRouter(prefix="/{network_id}/streams")


@router.get("/best")
def get_best_stream(
    network_id: str = Path(..., description="The ID of the network"),
    db: Session = Depends(db.get_db),
):
    """Get the currently selected best audio stream."""

    network = (
        db.query(models.MiraNetwork)
        .filter(models.MiraNetwork.id == uuid.UUID(network_id))
        .first()
    )
    if not network:
        raise HTTPException(status_code=404, detail="Network not found")

    multi_stream_processor = get_multi_stream_processor(network_id)
    best_stream_info = multi_stream_processor.get_best_stream()

    if not best_stream_info or not best_stream_info.get("client_id"):
        return {"best_stream": None}

    return {"best_stream": best_stream_info}


@router.get("/scores")
def get_all_stream_scores(
    network_id: str = Path(..., description="The ID of the network"),
    db: Session = Depends(db.get_db),
):
    """Get quality scores for all active streams."""

    network = (
        db.query(models.MiraNetwork)
        .filter(models.MiraNetwork.id == uuid.UUID(network_id))
        .first()
    )
    if not network:
        raise HTTPException(status_code=404, detail="Network not found")

    try:
        multi_stream_processor = get_multi_stream_processor(network_id)
        clients_info = multi_stream_processor.get_all_clients()
        scores = multi_stream_processor.get_all_stream_scores()
        best_stream = multi_stream_processor.get_best_stream()

        return {
            "active_streams": len(clients_info),
            "stream_scores": scores,
            "current_best": best_stream,
        }

    except Exception as e:
        MiraLogger.error(f"Error getting stream scores: {e}")
        raise HTTPException(
            status_code=500, detail=f"Failed to get stream scores: {str(e)}"
        )


@router.get("/{client_id}/info")
def get_client_stream_info(
    client_id: str = Path(..., description="The ID of the client"),
    network_id: str = Path(..., description="The ID of the network"),
    db: Session = Depends(db.get_db),
):
    """Get detailed stream information about a specific client."""

    network = (
        db.query(models.MiraNetwork)
        .filter(models.MiraNetwork.id == uuid.UUID(network_id))
        .first()
    )
    if not network:
        raise HTTPException(status_code=404, detail="Network not found")

    if client_id not in network.connected_clients:
        raise HTTPException(
            status_code=404, detail=f"Client {client_id} not found in network"
        )

    try:
        multi_stream_processor = get_multi_stream_processor(network_id)
        client_info = multi_stream_processor.get_client_info(client_id)
        current_score = multi_stream_processor.get_all_stream_scores().get(
            client_id, 0.0
        )
        best_stream = multi_stream_processor.get_best_stream()
        is_best_stream = best_stream and best_stream.get("client_id") == client_id

        return {
            "client_id": client_id,
            "quality_metrics": client_info.get("quality_metrics", {}),
            "current_score": round(current_score, 2),
            "is_best_stream": is_best_stream,
            "last_update": client_info.get("last_update"),
        }
    except Exception as e:
        MiraLogger.error(f"Error getting client stream info: {e}")
        raise HTTPException(
            status_code=500, detail=f"Failed to get client stream info: {str(e)}"
        )


@router.post("/phone/location")
def update_phone_location(
    request: dict = Body(...),
    network_id: str = Path(..., description="The ID of the network"),
    db: Session = Depends(db.get_db),
):
    """Update GPS-based location data for phone tracking."""

    network = (
        db.query(models.MiraNetwork)
        .filter(models.MiraNetwork.id == uuid.UUID(network_id))
        .first()
    )
    if not network:
        raise HTTPException(status_code=404, detail="Network not found")

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
                raise HTTPException(
                    status_code=400, detail=f"location.{field} is required"
                )

        multi_stream_processor = get_multi_stream_processor(network_id)
        success = multi_stream_processor.set_phone_location(client_id, location)

        if not success:
            raise HTTPException(status_code=404, detail=f"Client {client_id} not found")

        MiraLogger.info(f"Updated phone location for {client_id}: {location}")

        return {
            "message": f"Phone location updated successfully for {client_id}",
            "location": location,
        }

    except HTTPException:
        raise
    except Exception as e:
        MiraLogger.error(f"Error updating phone location: {e}")
        raise HTTPException(
            status_code=500, detail=f"Failed to update phone location: {str(e)}"
        )


@router.post("/phone/rssi")
def update_phone_rssi(
    request: dict = Body(...),
    network_id: str = Path(..., description="The ID of the network"),
    db: Session = Depends(db.get_db),
):
    """Update RSSI-based proximity data for phone tracking to specific client."""

    network = (
        db.query(models.MiraNetwork)
        .filter(models.MiraNetwork.id == uuid.UUID(network_id))
        .first()
    )
    if not network:
        raise HTTPException(status_code=404, detail="Network not found")

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

        multi_stream_processor = get_multi_stream_processor(network_id)
        success = multi_stream_processor.set_phone_rssi(target_client_id, rssi)

        if not success:
            raise HTTPException(
                status_code=404, detail=f"Target client {target_client_id} not found"
            )

        MiraLogger.info(
            f"Updated RSSI from {phone_client_id} to {target_client_id}: {rssi} dBm"
        )

        return {
            "message": f"RSSI updated successfully from {phone_client_id} to {target_client_id}",
            "rssi": rssi,
        }

    except HTTPException:
        raise
    except Exception as e:
        MiraLogger.error(f"Error updating phone RSSI: {e}")
        raise HTTPException(
            status_code=500, detail=f"Failed to update phone RSSI: {str(e)}"
        )
