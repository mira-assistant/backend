"""
Task management endpoints.
"""
from typing import List
from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session
import uuid

from app.api.deps import get_db_dependency, get_current_user_dependency
from app.schemas.action import Action, ActionCreate, ActionUpdate
from app.models.action import Action as ActionModel

router = APIRouter()


@router.post("/", response_model=Action)
def create_action(
    action: ActionCreate,
    db: Session = Depends(get_db_dependency),
    current_user: dict = Depends(get_current_user_dependency)
):
    """Create a new action/task."""
    db_action = ActionModel(**action.dict())
    db.add(db_action)
    db.commit()
    db.refresh(db_action)
    return db_action


@router.get("/", response_model=List[Action])
def get_actions(
    skip: int = 0,
    limit: int = 100,
    db: Session = Depends(get_db_dependency),
    current_user: dict = Depends(get_current_user_dependency)
):
    """Get all actions for the current user."""
    actions = db.query(ActionModel).filter(
        ActionModel.user_id == current_user["user_id"]
    ).offset(skip).limit(limit).all()
    return actions


@router.get("/{action_id}", response_model=Action)
def get_action(
    action_id: str,
    db: Session = Depends(get_db_dependency),
    current_user: dict = Depends(get_current_user_dependency)
):
    """Get a specific action by ID."""
    action = db.query(ActionModel).filter(
        ActionModel.id == uuid.UUID(action_id),
        ActionModel.user_id == current_user["user_id"]
    ).first()

    if not action:
        raise HTTPException(status_code=404, detail="Action not found")

    return action


@router.put("/{action_id}", response_model=Action)
def update_action(
    action_id: str,
    action_update: ActionUpdate,
    db: Session = Depends(get_db_dependency),
    current_user: dict = Depends(get_current_user_dependency)
):
    """Update an action."""
    action = db.query(ActionModel).filter(
        ActionModel.id == uuid.UUID(action_id),
        ActionModel.user_id == current_user["user_id"]
    ).first()

    if not action:
        raise HTTPException(status_code=404, detail="Action not found")

    for field, value in action_update.dict(exclude_unset=True).items():
        setattr(action, field, value)

    db.commit()
    db.refresh(action)
    return action


@router.delete("/{action_id}")
def delete_action(
    action_id: str,
    db: Session = Depends(get_db_dependency),
    current_user: dict = Depends(get_current_user_dependency)
):
    """Delete an action."""
    action = db.query(ActionModel).filter(
        ActionModel.id == uuid.UUID(action_id),
        ActionModel.user_id == current_user["user_id"]
    ).first()

    if not action:
        raise HTTPException(status_code=404, detail="Action not found")

    db.delete(action)
    db.commit()

    return {"detail": "Action deleted successfully"}

