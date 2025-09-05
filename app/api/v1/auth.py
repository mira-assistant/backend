"""
Authentication endpoints.
"""

from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session

from app.api.deps import get_db_dependency
from app.models.network import MiraNetwork
from app.schemas.network import Network, NetworkCreate, NetworkAuth

router = APIRouter()


@router.post("/register", response_model=Network)
async def register_network(
    network: NetworkCreate,
    db: Session = Depends(get_db_dependency)
):
    """Register a new network."""
    db_network = MiraNetwork.create(
        name=network.name,
        password=network.password.get_secret_value()
    )

    db.add(db_network)
    db.commit()
    db.refresh(db_network)

    return db_network


@router.post("/login", response_model=Network)
async def login_network(
    auth: NetworkAuth,
    db: Session = Depends(get_db_dependency)
):
    """Login to a network."""
    network = db.query(MiraNetwork).filter(MiraNetwork.network_id == auth.network_id).first()
    if not network:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Network not found"
        )

    if not network.verify_password(auth.password.get_secret_value()):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid password"
        )

    try:
        if not bool(network.is_active):
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Network is inactive"
            )
    except (TypeError, ValueError):
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Error checking network status"
        )

    return network