"""
Context service for managing conversation context.
"""

from typing import Tuple
from sqlalchemy.orm import Session
from datetime import datetime, timezone

from app.models.interaction import Interaction


class ContextProcessor:
    """Processes and manages conversation context."""

    def __init__(self):
        """Initialize context processor."""
        self.current_context = ""

    def build_context(
        self,
        interaction: Interaction,
        db: Session,
        window_size: int = 5
    ) -> Tuple[str, bool]:
        """Build context from recent interactions."""
        # Get recent interactions from same network
        recent_interactions = (
            db.query(Interaction)
            .filter(Interaction.network_id == interaction.network_id)
            .order_by(Interaction.created_at.desc())
            .limit(window_size)
            .all()
        )

        # Build context string
        context_parts = []
        for inter in reversed(recent_interactions):
            context_parts.append(f"User: {inter.text}")

        context = "\n".join(context_parts)
        has_intent = self._detect_intent(interaction.text)

        return context, has_intent

    def _detect_intent(self, text: str) -> bool:
        """Simple intent detection."""
        # For now, just check if text is not empty
        return bool(text and text.strip())