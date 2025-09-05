"""
Task management endpoints.
"""

from typing import List
from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session
import uuid
from datetime import datetime, timezone

from app.api.deps import get_db_dependency, get_network_auth_dependency
from app.models.network import MiraNetwork
from app.schemas.action import Action, ActionCreate, ActionUpdate
from app.models.action import Action as ActionModel

router = APIRouter()


@router.post("/networks/{network_id}/actions", response_model=Action)
async def create_action(
    network_id: str,
    password: str,
    action: ActionCreate,
    network: MiraNetwork = Depends(get_network_auth_dependency()),
    db: Session = Depends(get_db_dependency)
):
    """Create a new action/task."""
    db_action = ActionModel(
        **action.dict(),
        network_id=network.id
    )
    db.add(db_action)
    db.commit()
    db.refresh(db_action)
    return db_action


@router.get("/networks/{network_id}/actions", response_model=List[Action])
async def get_actions(
    network_id: str,
    password: str,
    skip: int = 0,
    limit: int = 100,
    network: MiraNetwork = Depends(get_network_auth_dependency()),
    db: Session = Depends(get_db_dependency)
):
    """Get all actions for a network."""
    actions = (
        db.query(ActionModel)
        .filter(ActionModel.network_id == network.id)
        .offset(skip)
        .limit(limit)
        .all()
    )
    return actions


@router.get("/networks/{network_id}/actions/{action_id}", response_model=Action)
async def get_action(
    network_id: str,
    action_id: str,
    password: str,
    network: MiraNetwork = Depends(get_network_auth_dependency()),
    db: Session = Depends(get_db_dependency)
):
    """Get a specific action by ID."""
    action = (
        db.query(ActionModel)
        .filter(
            ActionModel.id == uuid.UUID(action_id),
            ActionModel.network_id == network.id
        )
        .first()
    )

    if not action:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Action not found"
        )

    return action


@router.put("/networks/{network_id}/actions/{action_id}", response_model=Action)
async def update_action(
    network_id: str,
    action_id: str,
    action_update: ActionUpdate,
    password: str,
    network: MiraNetwork = Depends(get_network_auth_dependency()),
    db: Session = Depends(get_db_dependency)
):
    """Update an action."""
    action = (
        db.query(ActionModel)
        .filter(
            ActionModel.id == uuid.UUID(action_id),
            ActionModel.network_id == network.id
        )
        .first()
    )

    if not action:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Action not found"
        )

    # Update fields
    for field, value in action_update.dict(exclude_unset=True).items():
        try:
            if field == 'is_completed' and value and not bool(action.is_completed):
                setattr(action, 'completed_at', datetime.now(timezone.utc))
        except (TypeError, ValueError):
            pass
        setattr(action, field, value)

    db.commit()
    db.refresh(action)
    return action


@router.delete("/networks/{network_id}/actions/{action_id}")
async def delete_action(
    network_id: str,
    action_id: str,
    password: str,
    network: MiraNetwork = Depends(get_network_auth_dependency()),
    db: Session = Depends(get_db_dependency)
):
    """Delete an action."""
    action = (
        db.query(ActionModel)
        .filter(
            ActionModel.id == uuid.UUID(action_id),
            ActionModel.network_id == network.id
        )
        .first()
    )

    if not action:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Action not found"
        )

    db.delete(action)
    db.commit()

    return {"status": "success", "message": "Action deleted"}