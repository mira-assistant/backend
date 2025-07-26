"""
Context-aware action processor for Mira assistant.

This module processes transcribed text from whisper_live.py and determines actions
based on the context of the input, including both short-term and long-term context.

Features:
- Context storage and retrieval (short-term: last 10 seconds, long-term: historical)
- Intent classification with context enhancement
- Action parsing and detail extraction
- Modular design for easy integration with existing pipeline
"""

from __future__ import annotations

import re
import time
from collections import defaultdict
from datetime import datetime
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, asdict


@dataclass
class Interaction:
    """Represents a single interaction with timestamp and metadata."""

    timestamp: float
    formatted_timestamp: str
    speaker: str
    text: str
    raw_input: str

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class ActionDetails:
    """Structured representation of extracted action details."""

    action_type: Optional[str] = None
    call_to_action: bool = False

    # Contact details
    contact_method: Optional[str] = None
    contact_recipient: Optional[str] = None
    contact_message: Optional[str] = None

    # Reminder details
    remind_task: Optional[str] = None
    remind_time: Optional[str] = None

    # Schedule details
    schedule_event: Optional[str] = None
    schedule_time: Optional[str] = None
    schedule_duration: Optional[str] = None
    schedule_location: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {k: v for k, v in asdict(self).items() if v is not None}


class ContextProcessor:
    """
    Main context processor for handling transcribed text and determining actions.
    """

    def __init__(
        self,
        short_term_window: int = 10,
        max_history_size: int = 1000,
        max_conversation_length: int = 20,
        conversation_start: Optional[str] = None
    ):
        """
        Initialize the context processor.

        Args:
            short_term_window: Time window in seconds for short-term context (used as fallback)
            max_history_size: Maximum number of interactions to keep in history
            max_conversation_length: Maximum number of interactions to include in short-term context
        """
        self.short_term_window = short_term_window
        self.max_history_size = max_history_size
        self.max_conversation_length = max_conversation_length
        self.interaction_history: List[Interaction] = []
        self.keyword_index: Dict[str, List[int]] = defaultdict(
            list
        )

    def parse_whisper_output(self, whisper_output: str) -> Optional[Interaction]:
        """
        Parse the output from whisper_live.py process_sentence() function.

        Expected format: "(timestamp) Person X: text"

        Args:
            whisper_output: Raw output from whisper_live.py

        Returns:
            Parsed Interaction object or None if parsing fails
        """
        if not whisper_output or not whisper_output.strip():
            return None

        # Pattern to match: (YYYY-MM-DD HH:MM:SS) Person X: text
        pattern = r"\((\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2})\) (Person \d+): (.+)"
        match = re.match(pattern, whisper_output.strip())

        if not match:
            return None

        formatted_timestamp, speaker, text = match.groups()

        # Convert to Unix timestamp
        try:
            dt = datetime.strptime(formatted_timestamp, "%Y-%m-%d %H:%M:%S")
            timestamp = dt.timestamp()
        except ValueError:
            timestamp = time.time()

        return Interaction(
            timestamp=timestamp,
            formatted_timestamp=formatted_timestamp,
            speaker=speaker,
            text=text.strip(),
            raw_input=whisper_output.strip(),
        )

    def add_interaction(self, interaction: Interaction) -> None:
        """
        Add a new interaction to the history and update keyword index.

        Args:
            interaction: Interaction object to add
        """
        self.interaction_history.append(interaction)

        # Update keyword index
        words = interaction.text.lower().split()
        interaction_index = len(self.interaction_history) - 1

        for word in words:
            # Clean word of punctuation
            clean_word = re.sub(r"[^\w]", "", word)
            if clean_word:
                self.keyword_index[clean_word].append(interaction_index)

        # Maintain history size limit
        if len(self.interaction_history) > self.max_history_size:
            self._cleanup_old_interactions()

    def _cleanup_old_interactions(self) -> None:
        """Remove old interactions to maintain history size limit."""
        excess_count = len(self.interaction_history) - self.max_history_size
        if excess_count > 0:
            # Remove oldest interactions
            self.interaction_history = self.interaction_history[excess_count:]

            # Rebuild keyword index
            self.keyword_index.clear()
            for i, interaction in enumerate(self.interaction_history):
                words = interaction.text.lower().split()
                for word in words:
                    clean_word = re.sub(r"[^\w]", "", word)
                    if clean_word:
                        self.keyword_index[clean_word].append(i)

    def get_short_term_context(self, current_time: float) -> List[Interaction]:
        """
        Retrieve interactions from the start of the current 2-person conversation,
        up to a maximum length.

        Args:
            current_time: Current timestamp

        Returns:
            List of conversation interactions
        """
        if not self.interaction_history:
            return []

        # Find the current interaction index (most recent with timestamp <= current_time)
        current_index = -1
        for i in range(len(self.interaction_history) - 1, -1, -1):
            if self.interaction_history[i].timestamp <= current_time:
                current_index = i
                break

        if current_index == -1:
            return []

        # Find the start of the conversation
        conversation_start = self.find_conversation_start(current_index)

        # Get conversation interactions up to max length
        conversation_interactions = self.interaction_history[
            conversation_start : current_index + 1
        ]

        # Limit to max_conversation_length, keeping the most recent ones
        if len(conversation_interactions) > self.max_conversation_length:
            conversation_interactions = conversation_interactions[
                -self.max_conversation_length :
            ]

        return conversation_interactions

    def find_conversation_start(self, current_index: int) -> int:
        """
        Find the start of a 2-person conversation working backwards from current_index.

        Args:
            current_index: Index of the current interaction in history

        Returns:
            Index of the conversation start
        """
        if current_index < 0 or current_index >= len(self.interaction_history):
            return max(0, current_index)

        # Get all speakers in recent history
        speakers = set()
        conversation_start = current_index

        # Work backwards to find conversation participants and start
        for i in range(current_index, -1, -1):
            interaction = self.interaction_history[i]
            speakers.add(interaction.speaker)

            # If we have more than 2 speakers, the conversation started after this point
            if len(speakers) > 2:
                conversation_start = i + 1
                break

            # Check for conversation gap (more than 5 minutes between interactions)
            if i > 0:
                prev_interaction = self.interaction_history[i - 1]
                time_gap = interaction.timestamp - prev_interaction.timestamp
                if time_gap > 300:  # 5 minutes gap indicates conversation boundary
                    conversation_start = i
                    break

            conversation_start = i

        return conversation_start

    def get_long_term_context(
        self, keywords: List[str], max_results: int = 5
    ) -> List[Interaction]:
        """
        Retrieve relevant historical interactions based on keywords.

        Args:
            keywords: List of keywords to search for
            max_results: Maximum number of results to return

        Returns:
            List of relevant historical interactions
        """
        if not keywords:
            return []

        # Score interactions based on keyword matches
        interaction_scores: Dict[int, int] = defaultdict(int)

        for keyword in keywords:
            clean_keyword = re.sub(r"[^\w]", "", keyword.lower())
            if clean_keyword in self.keyword_index:
                for interaction_idx in self.keyword_index[clean_keyword]:
                    interaction_scores[interaction_idx] += 1

        # Sort by score (descending) and then by recency (descending)
        sorted_interactions = sorted(
            interaction_scores.items(),
            key=lambda x: (x[1], self.interaction_history[x[0]].timestamp),
            reverse=True,
        )

        # Return top results
        result_indices = [idx for idx, _ in sorted_interactions[:max_results]]
        return [self.interaction_history[idx] for idx in result_indices]

    def extract_keywords(self, text: str) -> List[str]:
        """
        Extract relevant keywords from text for context retrieval.

        Args:
            text: Input text

        Returns:
            List of keywords
        """
        # Simple keyword extraction - can be enhanced with NLP libraries
        words = text.lower().split()

        # Filter out common stop words
        stop_words = {
            "the",
            "a",
            "an",
            "and",
            "or",
            "but",
            "in",
            "on",
            "at",
            "to",
            "for",
            "of",
            "with",
            "by",
            "is",
            "am",
            "are",
            "was",
            "were",
            "be",
            "been",
            "being",
            "have",
            "has",
            "had",
            "do",
            "does",
            "did",
            "will",
            "would",
            "could",
            "should",
            "may",
            "might",
            "can",
            "i",
            "me",
            "my",
            "myself",
            "you",
            "your",
            "yours",
            "yourself",
            "he",
            "him",
            "his",
            "himself",
            "she",
            "her",
            "hers",
            "herself",
            "it",
            "its",
            "itself",
            "we",
            "us",
            "our",
            "ours",
            "ourselves",
            "they",
            "them",
            "their",
            "theirs",
            "themselves",
        }

        keywords = []
        for word in words:
            clean_word = re.sub(r"[^\w]", "", word)
            if clean_word and clean_word not in stop_words and len(clean_word) > 2:
                keywords.append(clean_word)

        return keywords

    def build_context_prompt(self, current_text: str, current_time: float) -> str:
        """
        Build an enhanced prompt with relevant context, excluding the current input.

        Args:
            current_text: Current input text
            current_time: Current timestamp

        Returns:
            Enhanced prompt with context
        """
        # Get short-term context, excluding the current input
        short_term = self.get_short_term_context(current_time)
        if short_term and current_text.__contains__(short_term[-1].text):
            short_term = short_term[:-1]

        # Extract keywords for long-term context
        keywords = self.extract_keywords(current_text)
        long_term = self.get_long_term_context(keywords)
        # Exclude current input from long-term context if present
        long_term = [i for i in long_term if i.text != current_text]

        # Build context sections
        context_parts = []

        if short_term:
            context_parts.append("Current conversation:")
            for interaction in short_term:
                context_parts.append("\n" + f"- {interaction.speaker}: {interaction.text}")

        if long_term:
            context_parts.append("\n\nRelevant historical context:")
            for interaction in long_term:
                context_parts.append(
                    "\n" + f"- {interaction.formatted_timestamp} {interaction.speaker}: {interaction.text}"
                )

        if context_parts:
            context_string = "".join(context_parts)
            return context_string
        
        return ""

    def classify_intent(self, text: str) -> bool:
        """
        Determine if the text contains actionable intent.

        Args:
            text: Input text to analyze

        Returns:
            True if actionable intent is detected, False otherwise
        """
        # Simple keyword-based intent detection
        action_keywords = {
            "contact": [
                "call",
                "text",
                "message",
                "email",
                "tell",
                "contact",
                "reach out",
            ],
            "remind": [
                "remind",
                "remember",
                "later",
                "tomorrow",
                "next week",
                "schedule reminder",
            ],
            "schedule": [
                "schedule",
                "appointment",
                "meeting",
                "book",
                "plan",
                "event",
                "calendar",
                "let's",
                "let us",
                "we'll",
                "we will",
                "we're",
                "we are",
                "we can",
                "we could",
                "we should",
                "we might",
            ],
            "generic": [
                "yes",
                "okay",
                "sure",
                "alright",
                "sounds good",
                "go ahead",
                "do it",
                "i'm",
                "i'll",
                "bet",
            ]
        }

        text_lower = text.lower()

        for category, keywords in action_keywords.items():
            if any(keyword in text_lower for keyword in keywords):
                return True

        return False

    def parse_action_details(
        self, text: str, llm_response: Dict[str, Any]
    ) -> ActionDetails:
        """
        Parse actionable details from text and LLM response.

        Args:
            text: Original input text
            llm_response: Response from the language model

        Returns:
            ActionDetails object with extracted information
        """
        action_details = ActionDetails()

        # Extract from LLM response if available
        if llm_response.get("call_to_action"):
            action_details.call_to_action = True
            action_details.action_type = llm_response.get("action_type")

            # Contact details
            if "contact" in llm_response:
                contact_info = llm_response["contact"]
                action_details.contact_method = contact_info.get("method", "message")
                action_details.contact_recipient = contact_info.get("recipient")
                action_details.contact_message = contact_info.get("message")

            # Reminder details
            if "remind" in llm_response:
                remind_info = llm_response["remind"]
                action_details.remind_task = remind_info.get("task")
                action_details.remind_time = remind_info.get("time")

            # Schedule details
            if "schedule" in llm_response:
                schedule_info = llm_response["schedule"]
                action_details.schedule_event = schedule_info.get("event")
                action_details.schedule_time = schedule_info.get("time")
                action_details.schedule_duration = schedule_info.get("duration")
                action_details.schedule_location = schedule_info.get("location")

        return action_details

    def process_input(self, whisper_output: str) -> Tuple[str, bool]:
        """
        Main processing function that handles transcribed text and determines context and intent.

        Args:
            whisper_output: Raw output from whisper_live.py

        Returns:
            Tuple containing (context, has_intent)
            - context: Context for prompt ready for LLM
            - has_intent: Boolean indicating if actionable intent was detected
        """
        # Parse whisper output
        interaction = self.parse_whisper_output(whisper_output)
        if not interaction:
            return whisper_output, False

        # Add to history
        self.add_interaction(interaction)

        # Build context-enhanced prompt
        enhanced_prompt = self.build_context_prompt(
            interaction.text, interaction.timestamp
        )

        # Classify intent
        has_intent = self.classify_intent(interaction.text)

        return enhanced_prompt, has_intent


# Utility functions for integration
def create_context_processor(max_conversation_length: int = 20) -> ContextProcessor:
    """Create a new context processor instance with default settings."""
    return ContextProcessor(max_conversation_length=max_conversation_length)


def process_whisper_output(
    processor: ContextProcessor, whisper_output: str
) -> Tuple[str, bool]:
    """
    Convenience function to process whisper output with context.

    Args:
        processor: ContextProcessor instance
        whisper_output: Raw output from whisper_live.py

    Returns:
        Tuple containing (enhanced_message, has_intent)
    """
    return processor.process_input(whisper_output)
