"""
Client registration schemas for request/response models.
"""

from typing import Optional
from datetime import datetime

from pydantic import BaseModel, HttpUrl


class ClientRegistrationRequest(BaseModel):
    """Schema for client registration request."""

    webhook_url: HttpUrl


class ClientRegistrationResponse(BaseModel):
    """Schema for client registration response."""

    message: str
    client_id: str
    webhook_url: str
    registered_at: datetime


class ClientDeregistrationResponse(BaseModel):
    """Schema for client deregistration response."""

    message: str
    client_id: str


class WebhookPayload(BaseModel):
    """Schema for webhook payload sent to clients."""

    interaction_id: str
    network_id: str
    text: str
    timestamp: datetime
    speaker_id: Optional[str] = None
    conversation_id: Optional[str] = None
