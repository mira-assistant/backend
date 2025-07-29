"""
Context Processor for Mira with advanced NLP features.

This module provides a comprehensive context processor that includes:
- Advanced speaker recognition with clustering
- Database integration with Person entities
- NLP features (NER, coreference resolution, topic modeling)
- Improved conversation management
- Context summarization
"""

from __future__ import annotations

import re
import json
import logging
from collections import defaultdict, deque
from datetime import datetime, timezone, timedelta
from typing import Dict, List, Optional, Tuple, Any, Set
import uuid
import numpy as np
from sklearn.cluster import DBSCAN
from sqlalchemy.orm import Session
from sqlalchemy import or_
import spacy
from transformers.pipelines import pipeline
from sentence_transformers import SentenceTransformer
from models import (
    Person,
    Interaction,
    Conversation,
)
from context_config import ContextProcessorConfig, DEFAULT_CONFIG
from db import get_db_session


# Removed custom dataclasses - now using SQLAlchemy models directly


class ContextProcessor:
    """
    Context processor with advanced NLP and speaker recognition features.
    """

    def __init__(self, config: Optional[ContextProcessorConfig] = None):
        """Initialize the context processor."""
        self.config = config or DEFAULT_CONFIG
        # Remove in-memory structures - all data now comes from database
        self.keyword_index: Dict[str, List[int]] = defaultdict(
            list
        )  # Keep for backward compatibility
        self.conversation_cache: deque = deque(maxlen=100)  # Keep for caching recent conversations

        # Clustering components
        self.dbscan = DBSCAN(
            eps=self.config.dbscan_eps,
            min_samples=self.config.dbscan_min_samples,
        )
        self.voice_embedding_cache: Dict[str, np.ndarray] = {}

        # Current conversation tracking for real-time updates
        self.current_conversation_id: Optional[uuid.UUID] = None
        self.current_participants: Set[str] = set()

        logging.basicConfig(level=getattr(logging, self.config.log_level))
        self.logger = logging.getLogger(__name__)

        self._init_nlp_components()

    def _init_nlp_components(self):
        """Initialize NLP components based on configuration."""
        self.nlp_components = {}

        try:
            if self.config.enable_ner or self.config.enable_coreference:
                try:
                    self.nlp_components["spacy"] = spacy.load(self.config.spacy_model)
                except OSError:
                    self.logger.warning(
                        f"spaCy model {self.config.spacy_model} not found, disabling NER/coreference"
                    )

            if self.config.enable_sentiment_analysis:
                self.nlp_components["sentiment"] = pipeline(
                    task="text-classification",
                    model="cardiffnlp/twitter-roberta-base-sentiment-latest",
                    top_k=None,
                )
            elif self.config.enable_sentiment_analysis:
                self.logger.warning("transformers not available, disabling sentiment analysis")

            if self.config.enable_topic_modeling:
                self.nlp_components["sentence_transformer"] = SentenceTransformer(
                    "all-MiniLM-L6-v2"
                )
            elif self.config.enable_topic_modeling:
                self.logger.warning("sentence-transformers not available, disabling topic modeling")

        except Exception as e:
            self.logger.error(f"Failed to initialize NLP components: {e}")
            self.nlp_components = {}

    def parse_whisper_output(self, whisper_output: str) -> Optional[Interaction]:
        """Parse whisper output and return SQLAlchemy Interaction model."""
        if not whisper_output or not whisper_output.strip():
            return None

        # Enhanced pattern to handle various formats
        patterns = [
            r"\((\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2})\) (Person \d+): (.+)",
            r"(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}) - (Person \d+): (.+)",
            r"\[(.*?)\] (Person \d+): (.+)",
        ]

        match = None
        for pattern in patterns:
            match = re.match(pattern, whisper_output.strip())
            if match:
                break

        if not match:
            # Fallback: try to extract just the text
            self.logger.warning(f"Could not parse whisper output format: {whisper_output}")
            return None

        formatted_timestamp, speaker, text = match.groups()

        # Convert to Unix timestamp
        try:
            dt = datetime.strptime(formatted_timestamp, "%Y-%m-%d %H:%M:%S")
            timestamp = dt.replace(tzinfo=timezone.utc)
        except ValueError:
            timestamp = datetime.now(timezone.utc)

        # Extract speaker index
        speaker_match = re.search(r"Person (\d+)", speaker)
        speaker_index = int(speaker_match.group(1)) if speaker_match else 1

        # Get or create person in database
        session = get_db_session()
        try:
            person = self.get_or_create_person(speaker_index, session)

            # Create SQLAlchemy Interaction object
            interaction = Interaction(
                text=text.strip(),
                timestamp=timestamp,
                speaker_id=person.id,
            )

            # Add NLP processing
            self._process_nlp_features(interaction)

            return interaction

        finally:
            session.close()

    def _process_nlp_features(self, interaction: Interaction):
        """Process NLP features for a SQLAlchemy interaction."""

        try:
            # Named Entity Recognition
            if self.config.enable_ner and "spacy" in self.nlp_components:
                doc = self.nlp_components["spacy"](interaction.text)
                setattr(
                    interaction,
                    "entities",
                    [
                        {
                            "text": ent.text,
                            "label": ent.label_,
                            "start": ent.start_char,
                            "end": ent.end_char,
                        }
                        for ent in doc.ents
                    ],
                )

            # Sentiment Analysis
            if self.config.enable_sentiment_analysis and "sentiment" in self.nlp_components:
                sentiment_result = self.nlp_components["sentiment"](interaction.text)
                if sentiment_result and len(sentiment_result[0]) > 0:
                    # Get the positive sentiment score
                    positive_score = next(
                        (
                            item["score"]
                            for item in sentiment_result[0]
                            if item["label"] == "LABEL_2"
                        ),
                        0.5,
                    )
                    setattr(interaction, "sentiment", positive_score)

            # Text Embedding for semantic similarity
            if (
                self.config.enable_context_summarization
                and "sentence_transformer" in self.nlp_components
            ):
                embedding = self.nlp_components["sentence_transformer"].encode(interaction.text)
                # Store embedding as JSON in voice_embedding field (reusing existing field)
                interaction.voice_embedding = embedding.tolist()

        except Exception as e:
            self.logger.error(f"NLP processing failed: {e}")

    def get_or_create_person(self, speaker_index: int, session: Session) -> Person:
        """Get or create a person entity in the database."""
        person = session.query(Person).filter_by(speaker_index=speaker_index).first()

        if not person:
            person = Person(
                speaker_index=speaker_index,
                name=f"Person {speaker_index}",
                is_identified=False,
            )
            session.add(person)
            session.commit()
            session.refresh(person)

        return person

    def assign_speaker(self, new_embedding: np.ndarray) -> str:
        """
        Assign embedding to a speaker; integrate robust method from sentence_processor.py.
        Returns person_id.
        """
        session = get_db_session()
        try:
            persons = session.query(Person).all()

            similarity_threshold = self.config.similarity_threshold

            for person in persons:
                if hasattr(person, "voice_embedding"):
                    voice_embedding = np.array(person.voice_embedding, dtype=np.float32)
                    similarity_score = self._cosine_sim(voice_embedding, new_embedding)

                    if similarity_score >= similarity_threshold:
                        # Weighted update: new_embedding = alpha * new_embedding + (1 - alpha) * old_embedding
                        # alpha decreases as number of interactions increases
                        num_interactions = len(person.interactions)
                        alpha = 1.0 / (num_interactions + 1)
                        updated_embedding = alpha * new_embedding + (1 - alpha) * voice_embedding
                        person.voice_embedding = updated_embedding.tolist()
                        session.commit()
                        return getattr(person, "id")  # Return UUID object, not string

            # If no existing speaker matches, create a new one
            max_speaker_index = (
                session.query(Person.speaker_index).order_by(Person.speaker_index.desc()).first()
            )
            next_speaker_index = (max_speaker_index[0] + 1) if max_speaker_index else 1

            new_person = Person(
                voice_embedding=new_embedding.tolist(),
                speaker_index=next_speaker_index,
                name=f"Person {next_speaker_index}",
            )
            session.add(new_person)
            session.commit()
            session.refresh(new_person)
            return getattr(new_person, "id")  # Return UUID object, not string

        finally:
            session.close()

    def _cosine_sim(self, a: np.ndarray, b: np.ndarray) -> float:
        """Cosine similarity between two vectors."""
        return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))

    def update_speaker_clustering(self, voice_embedding: np.ndarray, person_id):
        """Update speaker clustering with new voice embedding - now database-backed."""
        session = get_db_session()
        try:
            person = session.query(Person).filter_by(id=person_id).first()
            if not person:
                return

            # Update voice embedding using weighted average (already done in assign_speaker)
            # Update clustering if we have enough data across all speakers
            persons_with_embeddings = (
                session.query(Person).filter(Person.voice_embedding.isnot(None)).all()
            )

            if len(persons_with_embeddings) >= self.config.dbscan_min_samples:
                self._update_clusters_db(session)

        finally:
            session.close()

    def _update_clusters_db(self, session):
        """Update DBSCAN clustering for all speakers using database."""
        persons = session.query(Person).filter(Person.voice_embedding.isnot(None)).all()

        if len(persons) < self.config.dbscan_min_samples:
            return

        try:
            all_embeddings = []
            person_ids = []

            for person in persons:
                all_embeddings.append(np.array(person.voice_embedding))
                person_ids.append(person.id)

            all_embeddings = np.array(all_embeddings)
            cluster_labels = self.dbscan.fit_predict(all_embeddings)

            # Update cluster IDs in database
            for i, person_id in enumerate(person_ids):
                person = session.query(Person).filter_by(id=person_id).first()
                if person:
                    person.cluster_id = int(cluster_labels[i]) if cluster_labels[i] != -1 else None

            session.commit()

        except Exception as e:
            self.logger.error(f"Clustering update failed: {e}")

    def add_interaction(
        self,
        interaction: Interaction,
        voice_embedding: Optional[np.ndarray] = None,
    ) -> None:
        """Add interaction with database integration and real-time conversation management."""

        session = get_db_session()
        try:
            # Handle speaker recognition if voice embedding provided
            if voice_embedding is not None:
                person_id = self.assign_speaker(voice_embedding)
                setattr(interaction, "speaker_id", person_id)
                # Update clustering
                self.update_speaker_clustering(voice_embedding, person_id)

            # Check if we need to start a new conversation or continue existing one
            if self.detect_conversation_boundary_db(interaction, session):
                self._start_new_conversation_db(interaction, session)
            else:
                # Assign to current conversation if exists
                if self.current_conversation_id:
                    setattr(interaction, "conversation_id", self.current_conversation_id)

            # Save interaction to database
            session.add(interaction)
            session.commit()
            session.refresh(interaction)

            # Update current conversation participants
            if hasattr(interaction, "speaker_id"):
                self.current_participants.add(getattr(interaction, "speaker_id"))

            self.logger.debug(
                f"Interaction added to database with ID: {getattr(interaction, 'id')}"
            )

        except Exception as e:
            session.rollback()
            self.logger.error(f"Database integration failed: {e}")
        finally:
            session.close()

    def _update_keyword_index(self, interaction: Interaction):
        """Update keyword index."""
        words = interaction.text.lower().split()
        interaction_index = len(self.interaction_history) - 1

        for word in words:
            clean_word = re.sub(r"[^\w]", "", word)
            if clean_word and len(clean_word) > 2:
                self.keyword_index[clean_word].append(interaction_index)

        # Add entities as keywords
        if hasattr(interaction, "entities"):
            for entity in interaction.entities:
                entity_text = entity["text"].lower().replace(" ", "_")
                self.keyword_index[entity_text].append(interaction_index)

    def detect_conversation_boundary_db(self, current_interaction: Interaction, session) -> bool:
        """Conversation boundary detection using database queries."""
        # Get last interaction from database
        last_interaction = session.query(Interaction).order_by(Interaction.timestamp.desc()).first()

        if not last_interaction:
            return True

        # Time gap detection - handle timezone differences
        current_ts = current_interaction.timestamp
        last_ts = last_interaction.timestamp

        # Ensure both timestamps are timezone-aware
        if current_ts.tzinfo is None:
            current_ts = current_ts.replace(tzinfo=timezone.utc)
        if last_ts.tzinfo is None:
            last_ts = last_ts.replace(tzinfo=timezone.utc)

        time_gap = (current_ts - last_ts).total_seconds()
        if time_gap > self.config.conversation_gap_threshold:
            return True

        # Speaker change with extended silence
        if (
            current_interaction.speaker_id != last_interaction.speaker_id
            and time_gap > self.config.conversation_gap_threshold / 2
        ):
            return True

        # Topic shift detection using embeddings if available
        if (
            self.config.enable_topic_modeling
            and current_interaction.voice_embedding is not None
            and last_interaction.voice_embedding is not None
        ):
            current_embedding = np.array(current_interaction.voice_embedding)
            last_embedding = np.array(last_interaction.voice_embedding)

            similarity = self._cosine_sim(current_embedding, last_embedding)

            if similarity < self.config.context_similarity_threshold:
                return True

        return False

    def get_short_term_context(self, current_time: datetime) -> List[Interaction]:
        """Short-term context retrieval from database."""
        session = get_db_session()
        try:
            # Get interactions from current conversation or recent interactions
            if self.current_conversation_id:
                # Get interactions from current conversation
                interactions = (
                    session.query(Interaction)
                    .filter_by(conversation_id=self.current_conversation_id)
                    .order_by(Interaction.timestamp.desc())
                    .limit(self.config.max_conversation_length)
                    .all()
                )
                interactions.reverse()  # Return in chronological order
            else:
                # Get recent interactions within time window
                time_threshold = current_time - timedelta(
                    seconds=self.config.conversation_gap_threshold
                )
                interactions = (
                    session.query(Interaction)
                    .filter(Interaction.timestamp >= time_threshold)
                    .filter(Interaction.timestamp <= current_time)
                    .order_by(Interaction.timestamp.asc())
                    .limit(self.config.max_conversation_length)
                    .all()
                )

            return interactions

        finally:
            session.close()

    # Removed obsolete _find_conversation_start method (replaced by database queries)

    def get_long_term_context(
        self,
        keywords: List[str],
        current_interaction: Optional[Interaction] = None,
        max_results: Optional[int] = None,
    ) -> List[Interaction]:
        """Long-term context retrieval with semantic similarity from database."""
        max_results = max_results or self.config.long_term_context_max_results
        session = get_db_session()

        try:
            relevant_interactions = []

            # Keyword-based retrieval
            if keywords:
                keyword_results = self._get_keyword_interactions_db(keywords, session, max_results)
                relevant_interactions.extend(keyword_results)

            # Semantic similarity retrieval using embeddings
            if (
                current_interaction
                and current_interaction.voice_embedding is not None
                and self.config.enable_context_summarization
            ):
                semantic_results = self._get_semantic_similar_interactions_db(
                    getattr(current_interaction, "voice_embedding"), session, max_results
                )
                relevant_interactions.extend(semantic_results)

            # Remove duplicates and sort by relevance/recency
            seen_ids = set()
            unique_interactions = []
            for interaction in relevant_interactions:
                if interaction.id not in seen_ids:
                    seen_ids.add(interaction.id)
                    unique_interactions.append(interaction)

            # Sort by timestamp (most recent first) and limit results
            unique_interactions.sort(key=lambda x: x.timestamp, reverse=True)
            return unique_interactions[:max_results]

        finally:
            session.close()

    def _get_keyword_interactions_db(
        self, keywords: List[str], session, max_results: int
    ) -> List[Interaction]:
        """Get interactions matching keywords from database."""
        if not keywords:
            return []

        # Simple text search for keywords
        keyword_filters = []
        for keyword in keywords:
            keyword_filters.append(Interaction.text.like(f"%{keyword}%"))

        interactions = (
            session.query(Interaction)
            .filter(or_(*keyword_filters))
            .order_by(Interaction.timestamp.desc())
            .limit(max_results * 2)  # Get more to allow for deduplication
            .all()
        )

        return interactions

    def _get_semantic_similar_interactions_db(
        self, query_embedding: List[float], session, max_results: int
    ) -> List[Interaction]:
        """Get semantically similar interactions using embeddings from database."""
        query_embedding_np = np.array(query_embedding)

        # Get all interactions with embeddings
        interactions_with_embeddings = (
            session.query(Interaction)
            .filter(Interaction.voice_embedding.isnot(None))
            .order_by(Interaction.timestamp.desc())
            .limit(100)  # Limit to recent interactions for performance
            .all()
        )

        similarities = []
        for interaction in interactions_with_embeddings:
            try:
                interaction_embedding = np.array(interaction.voice_embedding)
                similarity = self._cosine_sim(query_embedding_np, interaction_embedding)
                if similarity >= self.config.context_similarity_threshold:
                    similarities.append((interaction, similarity))
            except Exception as e:
                self.logger.debug(f"Error computing similarity: {e}")

        # Sort by similarity and return top results
        similarities.sort(key=lambda x: x[1], reverse=True)
        return [interaction for interaction, _ in similarities[:max_results]]

    # Removed obsolete in-memory scoring methods (replaced by database queries)

    def build_context_prompt(self, current_text: str, current_time: datetime) -> str:
        """Build context prompt with summarization using database."""
        # Get contexts from database
        short_term = self.get_short_term_context(current_time)

        # Remove current interaction if it's already in short term context
        if short_term and current_text in short_term[-1].text:
            short_term = short_term[:-1]

        keywords = self.extract_keywords(current_text)

        # For long-term context, try to get current interaction for semantic similarity
        current_interaction = None
        session = get_db_session()
        try:
            current_interaction = (
                session.query(Interaction).order_by(Interaction.timestamp.desc()).first()
            )
        finally:
            session.close()

        long_term = self.get_long_term_context(keywords, current_interaction)
        long_term = [i for i in long_term if getattr(i, "text") != current_text]

        # Build context with summarization
        context_parts = []

        if short_term:
            context_parts.append("Current conversation:")
            for interaction in short_term:
                speaker_info = f"Person {self._get_speaker_index(interaction.speaker_id)}"
                if hasattr(interaction, "entities") and isinstance(interaction.entities, list):
                    entity_text = ", ".join(
                        [
                            str(e.get("text", ""))
                            for e in interaction.entities[:3]
                            if isinstance(e, dict) and "text" in e
                        ]
                    )
                    speaker_info += f" [entities: {entity_text}]"

                context_parts.append(f"\n- {speaker_info}: {interaction.text}")

        if long_term:
            if self.config.enable_context_summarization and len(long_term) > 3:
                # Summarize long-term context
                summary = self._summarize_context_db(long_term)
                context_parts.append(f"\n\nRelevant context summary: {summary}")
            else:
                context_parts.append("\n\nRelevant historical context:")
                for interaction in long_term:
                    timestamp_str = interaction.timestamp.strftime("%Y-%m-%d %H:%M:%S")
                    speaker_info = f"Person {self._get_speaker_index(interaction.speaker_id)}"
                    context_parts.append(f"\n- {timestamp_str} {speaker_info}: {interaction.text}")

        return "".join(context_parts) if context_parts else ""

    def _get_speaker_index(self, speaker_id) -> int:
        """Get speaker index from person ID."""
        if not speaker_id:
            return 1

        session = get_db_session()
        try:
            person = session.query(Person).filter_by(id=speaker_id).first()
            return getattr(person, "speaker_index") if person else 1
        finally:
            session.close()

    def _summarize_context_db(self, interactions: List[Interaction]) -> str:
        """Summarize context interactions from database models."""
        if not interactions:
            return ""

        # Simple extractive summarization
        texts = [getattr(i, "text") for i in interactions]

        # Combine entities and key phrases
        key_info = []
        for interaction in interactions:
            if hasattr(interaction, "entities"):
                entities = [
                    e["text"]
                    for e in interaction.entities
                    if e["label"] in ["PERSON", "ORG", "EVENT"]
                ]
                key_info.extend(entities)

        # Create summary
        combined_text = " ".join([str(t) for t in texts])
        sentences = combined_text.split(". ")

        # Keep the most informative sentences (simple heuristic)
        important_sentences = []
        for sentence in sentences[:5]:  # Limit to first 5 sentences
            if any(keyword in sentence.lower() for keyword in key_info[:10]):
                important_sentences.append(sentence)

        summary = ". ".join(important_sentences[:3])  # Max 3 sentences

        if len(summary) > self.config.summary_max_length:
            summary = summary[: self.config.summary_max_length] + "..."

        return summary or "Previous discussion about relevant topics."

    def extract_keywords(self, text: str) -> List[str]:
        """Keyword extraction with NLP features."""
        # Base keyword extraction (existing logic)
        words = text.lower().split()
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

        return list(set(keywords))  # Remove duplicates

    def classify_intent(self, text: str) -> bool:
        """Intent classification with NLP features."""
        # Base classification (existing logic)
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
            ],
        }

        text_lower = text.lower()
        has_keywords = any(
            any(keyword in text_lower for keyword in keywords)
            for keywords in action_keywords.values()
        )

        if has_keywords:
            return True

        return False

    def process_input(
        self, whisper_output: str, voice_embedding: Optional[np.ndarray] = None
    ) -> Tuple[str, bool]:
        """Input processing with full feature integration using database-only approach."""
        # Parse whisper output to get SQLAlchemy model
        interaction = self.parse_whisper_output(whisper_output)
        if not interaction:
            return whisper_output, False

        # Check for conversation boundary using database
        session = get_db_session()
        try:
            if self.detect_conversation_boundary_db(interaction, session):
                self._start_new_conversation_db(interaction, session)

            # Add to database with database integration
            self.add_interaction(interaction, voice_embedding)

            # Build context using database
            enhanced_prompt = self.build_context_prompt(
                getattr(interaction, "text"), getattr(interaction, "timestamp")
            )

            # Enhanced intent classification
            has_intent = self.classify_intent(getattr(interaction, "text"))

            if self.config.debug_mode:
                self.logger.debug(f"Processed interaction: {interaction.text}")
                self.logger.debug(f"Intent detected: {has_intent}")
                self.logger.debug(f"Entities: {interaction.entities}")

            return enhanced_prompt, has_intent

        finally:
            session.close()

    def _start_new_conversation_db(self, interaction: Interaction, session):
        """Start a new conversation in the database with real-time tracking."""
        try:
            # Create new conversation
            conversation = Conversation(
                user_ids=(
                    getattr(interaction, "speaker_id")
                    if getattr(interaction, "speaker_id")
                    else uuid.uuid4()
                ),
                speaker_id=getattr(interaction, "speaker_id"),
                start_of_conversation=getattr(interaction, "timestamp"),
                participants=json.dumps(
                    [str(getattr(interaction, "speaker_id"))]
                    if getattr(interaction, "speaker_id")
                    else []
                ),
            )

            session.add(conversation)
            session.commit()
            session.refresh(conversation)

            self.current_conversation_id = getattr(conversation, "id")
            self.current_participants = (
                {getattr(interaction, "speaker_id")}
                if getattr(interaction, "speaker_id")
                else set()
            )

            # Assign this conversation to the interaction
            interaction.conversation_id = conversation.id

            self.logger.debug(f"Started new conversation with ID: {conversation.id}")

        except Exception as e:
            self.logger.error(f"Failed to start new conversation: {e}")

    # Removed obsolete cleanup methods (no longer needed with database storage)

    def get_speaker_summary(self) -> Dict[str, Any]:
        """Get summary of all tracked speakers from database."""
        session = get_db_session()
        try:
            persons = session.query(Person).all()
            summary = {}

            for person in persons:
                summary[f"Person {person.speaker_index}"] = {
                    "interaction_count": len(person.interactions),
                    "cluster_id": person.cluster_id,
                    "is_identified": getattr(person, "is_identified", False),
                    "name": person.name,
                    "person_id": str(person.id),
                }

            return summary
        finally:
            session.close()


# Utility functions for integration
def create_context_processor(
    config: Optional[ContextProcessorConfig] = None,
) -> ContextProcessor:
    """Create a new context processor instance."""
    return ContextProcessor(config)


def process_interaction(
    processor: ContextProcessor,
    interaction: Interaction,
    voice_embedding: Optional[np.ndarray] = None,
) -> Tuple[str, bool]:
    """Process a database interaction."""
    # Format the interaction as the processor expects
    timestamp_str = interaction.timestamp.strftime("%Y-%m-%d %H:%M:%S")
    speaker_index = processor._get_speaker_index(interaction.speaker_id)
    formatted_input = f"({timestamp_str}) Person {speaker_index}: {interaction.text}"
    return processor.process_input(formatted_input, voice_embedding)
