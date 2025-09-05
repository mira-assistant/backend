"""
Dependencies for FastAPI endpoints.
"""

from fastapi import Depends, HTTPException, status
from sqlalchemy.orm import Session

from app.db.session import get_db_session
from app.models.network import MiraNetwork


def get_db_dependency():
    """Get database session."""
    db = get_db_session()
    try:
        yield db
    finally:
        db.close()


def get_network_auth_dependency():
    """Factory for network authentication dependency."""
    async def verify_network_auth(
        network_id: str,
        password: str,
        db: Session = Depends(get_db_dependency)
    ) -> MiraNetwork:
        """Verify network authentication and return network."""
        network = db.query(MiraNetwork).filter(MiraNetwork.network_id == network_id).first()
        if not network:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Network not found"
            )

        if not network.verify_password(password):
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

    return verify_network_auth