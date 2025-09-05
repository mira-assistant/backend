"""
Network management endpoints.
"""

from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session

from app.api.deps import get_db_dependency
from app.models.network import MiraNetwork
from app.schemas.network import (
    Network,
    NetworkCreate,
    NetworkUpdate,
    NetworkAuth,
    NetworkPasswordUpdate,
)

router = APIRouter()


@router.post("/", response_model=Network)
async def create_network(network: NetworkCreate, db: Session = Depends(get_db_dependency)):
    """Create a new Mira network."""
    db_network = MiraNetwork.create(name=network.name, password=network.password.get_secret_value())

    db.add(db_network)
    db.commit()
    db.refresh(db_network)

    return db_network


@router.post("/auth", response_model=Network)
async def authenticate_network(auth: NetworkAuth, db: Session = Depends(get_db_dependency)):
    """Authenticate with a network."""
    network = db.query(MiraNetwork).filter(MiraNetwork.network_id == auth.network_id).first()
    if not network:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Network not found")

    if not network.verify_password(auth.password.get_secret_value()):
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid password")

    return network


@router.get("/{network_id}", response_model=Network)
async def get_network(network_id: str, password: str, db: Session = Depends(get_db_dependency)):
    """Get network details."""
    network = db.query(MiraNetwork).filter(MiraNetwork.network_id == network_id).first()
    if not network:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Network not found")

    if not network.verify_password(password):
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid password")

    return network


@router.put("/{network_id}", response_model=Network)
async def update_network(
    network_id: str,
    updated_network: NetworkUpdate,
    password: str,
    db: Session = Depends(get_db_dependency),
):
    """Update network settings."""
    network = db.query(MiraNetwork).filter(MiraNetwork.network_id == network_id).first()
    if not network:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Network not found")

    if not network.verify_password(password):
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid password")

    # Update fields
    if updated_network.name is not None:
        network.name = updated_network.name  # type: ignore
    if updated_network.service_enabled is not None:
        network.service_enabled = updated_network.service_enabled  # type: ignore
    if updated_network.network_settings is not None:
        network.network_settings.update(updated_network.network_settings)

    db.commit()
    db.refresh(network)

    return network


@router.post("/{network_id}/password", response_model=Network)
async def update_network_password(
    network_id: str,
    password_update: NetworkPasswordUpdate,
    db: Session = Depends(get_db_dependency),
):
    """Update network password."""
    network = db.query(MiraNetwork).filter(MiraNetwork.network_id == network_id).first()
    if not network:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Network not found")

    if not network.verify_password(password_update.current_password.get_secret_value()):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid current password"
        )

    network.update_password(password_update.new_password.get_secret_value())
    db.commit()

    return network


@router.delete("/{network_id}")
async def delete_network(network_id: str, password: str, db: Session = Depends(get_db_dependency)):
    """Delete a network."""
    network = db.query(MiraNetwork).filter(MiraNetwork.network_id == network_id).first()
    if not network:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Network not found")

    if not network.verify_password(password):
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid password")

    db.delete(network)
    db.commit()

    return {"status": "success", "message": "Network deleted"}
