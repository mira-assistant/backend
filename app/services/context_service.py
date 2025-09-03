"""
Context processing service for conversation management.
"""
from __future__ import annotations

import re
import json
import logging
from datetime import timezone, timedelta
from typing import List, Optional, Tuple
import numpy as np

import spacy
from transformers.pipelines import pipeline
from sentence_transformers import SentenceTransformer

from sqlalchemy import or_
from sqlalchemy.orm import Session

from app.models.person import Person
from app.models.interaction import Interaction
from app.models.conversation import Conversation
from app.core.config import settings
from app.db.session import get_db_session

from typing import Literal


class ContextProcessorConfig:
    """Configuration class for the context processor."""

    class NLPConfig:
        """Natural Language Processing parameters."""
        SPACY_MODEL: Literal["en_core_web_sm"] = "en_core_web_sm"
        CONTEXT_SIMILARITY_THRESHOLD: float = settings.context_similarity_threshold

    class ContextManagementParameters:
        """Parameters for managing context and conversation boundaries."""
        CONVERSATION_GAP_THRESHOLD: int = settings.conversation_gap_threshold
        SHORT_TERM_CONTEXT_MAX_RESULTS: Literal[20] = 20
        LONG_TERM_CONTEXT_MAX_RESULTS: Literal[5] = 5
        SUMMARY_MAX_LENGTH: Literal[200] = 200

    class PerformanceConfig:
        """Performance optimization parameters."""
        BATCH_PROCESSING: Literal[False] = False
        CACHE_EMBEDDINGS: Literal[True] = True
        ASYNC_PROCESSING: Literal[False] = False

    class DebugConfig:
        """Debugging and logging parameters."""
        DEBUG_MODE: Literal[False] = False
        LOG_LEVEL: Literal["INFO"] = "INFO"


class ContextProcessor:
    """
    Context processor with advanced NLP and speaker recognition features.
    """

    def __init__(self):
        """Initialize the context processor."""
        self.current_conversation = Conversation()
        self.current_participants = set()

        logging.basicConfig(level=getattr(logging, ContextProcessorConfig.DebugConfig.LOG_LEVEL))
        self.logger = logging.getLogger(__name__)

        self._init_nlp_components()
        logging.info("ContextProcessor initialized")

    def _init_nlp_components(self):
        """Initialize NLP models as individual state variables."""
        self.spacy_model = spacy.load(ContextProcessorConfig.NLPConfig.SPACY_MODEL)
        self.sentence_transformer = SentenceTransformer("all-MiniLM-L6-v2")
        self.sentiment_pipeline = pipeline(
            task="text-classification",
            model="cardiffnlp/twitter-roberta-base-sentiment-latest",
            top_k=None,
        )

    def _process_nlp_features(self, interaction: Interaction):
        """Process NLP features for a SQLAlchemy interaction."""
        try:
            # Named Entity Recognition
            doc = self.spacy_model(interaction.text)
            entities_list = [
                {
                    "text": ent.text,
                    "label": ent.label_,
                    "start": ent.start_char,
                    "end": ent.end_char,
                }
                for ent in doc.ents
            ]

            interaction.entities = json.dumps(entities_list)

            # Sentiment Analysis
            sentiment_result = self.sentiment_pipeline(interaction.text)
            if sentiment_result and len(sentiment_result[0]) > 0:
                positive_score = next(
                    (item["score"] for item in sentiment_result[0] if item["label"] == "LABEL_2"),
                    0.5,
                )
                interaction.sentiment = positive_score

            # Text Embedding for semantic similarity
            embedding = self.sentence_transformer.encode(interaction.text, show_progress_bar=False)
            interaction.text_embedding = embedding.tolist()

        except Exception as e:
            self.logger.error(f"NLP processing failed: {e}")

    def _cosine_sim(self, a: np.ndarray, b: np.ndarray) -> float:
        """Cosine similarity between two vectors."""
        return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))

    def detect_conversation_boundary(self, current_interaction: Interaction) -> bool:
        """Conversation boundary detection using database queries."""
        if (
            self.current_conversation.interactions is None
            or len(self.current_conversation.interactions) == 0
        ):
            return True

        last_interaction = self.current_conversation.interactions[-1]

        current_ts = current_interaction.timestamp
        last_ts = last_interaction.timestamp

        if current_ts.tzinfo is None:
            current_ts = current_ts.replace(tzinfo=timezone.utc)
        if last_ts.tzinfo is None:
            last_ts = last_ts.replace(tzinfo=timezone.utc)

        time_gap = (current_ts - last_ts).total_seconds()
        if time_gap > ContextProcessorConfig.ContextManagementParameters.CONVERSATION_GAP_THRESHOLD:
            return True

        if (
            current_interaction.speaker_id != last_interaction.speaker_id
            and time_gap
            > ContextProcessorConfig.ContextManagementParameters.CONVERSATION_GAP_THRESHOLD / 2
        ):
            return True

        current_doc = self.spacy_model(current_interaction.text)
        last_doc = self.spacy_model(last_interaction.text)

        try:
            topic_similarity = current_doc.similarity(last_doc)
        except Exception as e:
            self.logger.debug(f"spaCy similarity failed: {e}")
            topic_similarity = 1.0

        if topic_similarity < ContextProcessorConfig.NLPConfig.CONTEXT_SIMILARITY_THRESHOLD:
            return True

        return False

    def get_short_term_context(self, current_time, db: Session) -> List[Interaction]:
        """Short-term context retrieval from database."""
        try:
            if self.current_conversation.id is not None:
                interactions = (
                    db.query(Interaction)
                    .filter_by(conversation_id=self.current_conversation.id)
                    .order_by(Interaction.timestamp.desc())
                    .limit(
                        ContextProcessorConfig.ContextManagementParameters.SHORT_TERM_CONTEXT_MAX_RESULTS
                    )
                    .all()
                )
                interactions.reverse()
            else:
                time_threshold = current_time - timedelta(
                    seconds=ContextProcessorConfig.ContextManagementParameters.CONVERSATION_GAP_THRESHOLD
                )
                interactions = (
                    db.query(Interaction)
                    .filter(Interaction.timestamp >= time_threshold)
                    .filter(Interaction.timestamp <= current_time)
                    .order_by(Interaction.timestamp.asc())
                    .limit(
                        ContextProcessorConfig.ContextManagementParameters.SHORT_TERM_CONTEXT_MAX_RESULTS
                    )
                    .all()
                )

            return interactions
        except Exception as e:
            self.logger.error(f"Error getting short-term context: {e}")
            return []

    def get_long_term_context(
        self,
        keywords: List[str],
        current_interaction: Optional[Interaction] = None,
        max_results: Optional[int] = None,
        db: Session = None
    ) -> List[Interaction]:
        """Long-term context retrieval with semantic similarity from database."""
        max_results = (
            max_results
            or ContextProcessorConfig.ContextManagementParameters.LONG_TERM_CONTEXT_MAX_RESULTS
        )

        try:
            relevant_interactions = []

            if keywords:
                keyword_results = self._get_keyword_interactions_db(keywords, db, max_results)
                relevant_interactions.extend(keyword_results)

            if current_interaction and current_interaction.text_embedding is not None:
                semantic_results = self._get_semantic_similar_interactions_db(
                    current_interaction.text_embedding, db, max_results
                )
                relevant_interactions.extend(semantic_results)

            seen_ids = set()
            unique_interactions = []
            for interaction in relevant_interactions:
                if interaction.id not in seen_ids:
                    seen_ids.add(interaction.id)
                    unique_interactions.append(interaction)

            unique_interactions.sort(key=lambda x: x.timestamp, reverse=True)
            return unique_interactions[:max_results]
        except Exception as e:
            self.logger.error(f"Error getting long-term context: {e}")
            return []

    def _get_keyword_interactions_db(
        self, keywords: List[str], db: Session, max_results: int
    ) -> List[Interaction]:
        """Get interactions matching keywords from database."""
        if not keywords:
            return []

        keyword_filters = []
        for keyword in keywords:
            keyword_filters.append(Interaction.text.like(f"%{keyword}%"))

        interactions = (
            db.query(Interaction)
            .filter(or_(*keyword_filters))
            .order_by(Interaction.timestamp.desc())
            .limit(max_results * 2)
            .all()
        )

        return interactions

    def _get_semantic_similar_interactions_db(
        self, query_embedding, db: Session, max_results: int
    ) -> List[Interaction]:
        """Get semantically similar interactions using text embeddings from database."""
        query_embedding_np = np.array(query_embedding)

        interactions_with_embeddings = (
            db.query(Interaction)
            .filter(Interaction.text_embedding.isnot(None))
            .order_by(Interaction.timestamp.desc())
            .limit(100)
            .all()
        )

        similarities = []
        for interaction in interactions_with_embeddings:
            try:
                interaction_embedding = np.array(interaction.text_embedding)
                similarity = self._cosine_sim(query_embedding_np, interaction_embedding)
                if similarity >= ContextProcessorConfig.NLPConfig.CONTEXT_SIMILARITY_THRESHOLD:
                    similarities.append((interaction, similarity))
            except Exception as e:
                self.logger.debug(f"Error computing similarity: {e}")

        similarities.sort(key=lambda x: x[1], reverse=True)
        return [interaction for interaction, _ in similarities[:max_results]]

    def build_context_prompt(self, interaction: Interaction, db: Session) -> str:
        """Build context prompt with summarization using database."""
        short_term = self.get_short_term_context(interaction.timestamp, db)
        if short_term and interaction.text in short_term[-1].text:
            short_term = short_term[:-1]

        keywords = self._extract_keywords(interaction.text)
        long_term = self.get_long_term_context(keywords, interaction, db=db)
        long_term = [i for i in long_term if str(i.text) != interaction.text]

        context_parts = []

        if short_term:
            context_parts.append("Current conversation:")
            for interaction in short_term:
                speaker_info = f"Person {self._get_speaker_index(interaction.speaker_id, db)}"
                if interaction.entities is not None and isinstance(interaction.entities, list):
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
            if len(long_term) > 3:
                summary = self._summarize_context(long_term)
                context_parts.append(f"\n\nRelevant context summary: {summary}")
            else:
                context_parts.append("\n\nRelevant historical context:")
                for interaction in long_term:
                    timestamp_str = interaction.timestamp.strftime("%Y-%m-%d %H:%M:%S")
                    speaker_info = f"Person {self._get_speaker_index(interaction.speaker_id, db)}"
                    context_parts.append(f"\n- {timestamp_str} {speaker_info}: {interaction.text}")

        return "".join(context_parts) if context_parts else ""

    def _get_speaker_index(self, speaker_id, db: Session):
        """Get speaker index from person ID."""
        try:
            person = db.query(Person).filter_by(id=speaker_id).first()
            if not person:
                raise ValueError(f"Person with ID {speaker_id} not found in database.")
            return person.index
        except Exception as e:
            self.logger.error(f"Error getting speaker index: {e}")
            return 0

    def _summarize_context(self, interactions: List[Interaction]) -> str:
        """Summarize context interactions from database models."""
        if not interactions:
            return ""

        texts = [i.text for i in interactions]

        key_info = []
        for interaction in interactions:
            if interaction.entities is not None:
                entities = [
                    e["text"]
                    for e in interaction.entities
                    if e["label"] in ["PERSON", "ORG", "EVENT"]
                ]
                key_info.extend(entities)

        combined_text = " ".join([str(t) for t in texts])
        sentences = combined_text.split(". ")

        important_sentences = []
        for sentence in sentences[:5]:
            if any(keyword in sentence.lower() for keyword in key_info[:10]):
                important_sentences.append(sentence)

        summary = ". ".join(important_sentences[:3])

        if len(summary) > ContextProcessorConfig.ContextManagementParameters.SUMMARY_MAX_LENGTH:
            summary = (
                summary[: ContextProcessorConfig.ContextManagementParameters.SUMMARY_MAX_LENGTH]
                + "..."
            )

        return summary or "Previous discussion about relevant topics."

    def _extract_keywords(self, text) -> List[str]:
        """Keyword extraction with NLP features."""
        words = str(text).lower().split()
        stop_words = {
            "the", "a", "an", "and", "or", "but", "in", "on", "at", "to", "for", "of", "with", "by",
            "is", "am", "are", "was", "were", "be", "been", "being", "have", "has", "had", "do", "does", "did",
            "will", "would", "could", "should", "may", "might", "can", "i", "me", "my", "myself",
            "you", "your", "yours", "yourself", "he", "him", "his", "himself", "she", "her", "hers", "herself",
            "it", "its", "itself", "we", "us", "our", "ours", "ourselves", "they", "them", "their", "theirs", "themselves",
        }

        keywords = []
        for word in words:
            clean_word = re.sub(r"[^\w]", "", word)
            if clean_word and clean_word not in stop_words and len(clean_word) > 2:
                keywords.append(clean_word)

        return list(set(keywords))

    def _classify_intent(self, text) -> bool:
        """Intent classification with NLP features."""
        action_keywords = {
            "contact": ["call", "text", "message", "email", "tell", "contact", "reach out"],
            "remind": ["remind", "remember", "later", "tomorrow", "next week", "schedule reminder"],
            "schedule": ["schedule", "appointment", "meeting", "book", "plan", "event", "calendar", "let's", "let us", "we'll", "we will", "we're", "we are", "we can", "we could", "we should", "we might"],
            "generic": ["yes", "okay", "sure", "alright", "sounds good", "go ahead", "do it", "i'm", "i'll", "bet"],
        }

        text_lower = str(text).lower()
        has_keywords = any(
            any(keyword in text_lower for keyword in keywords)
            for keywords in action_keywords.values()
        )

        return has_keywords

    def build_context(self, interaction: Interaction, db: Session) -> Tuple[str, bool]:
        """Interaction processing with full feature integration using database-only approach."""
        try:
            if self.detect_conversation_boundary(interaction):
                self.current_conversation = Conversation(user_ids=[interaction.speaker_id])

            self._process_nlp_features(interaction)
            enhanced_prompt = self.build_context_prompt(interaction, db)
            has_intent = self._classify_intent(interaction.text)

            if ContextProcessorConfig.DebugConfig.DEBUG_MODE:
                self.logger.debug(f"Processed interaction: {interaction.text}")
                self.logger.debug(f"Intent detected: {has_intent}")
                self.logger.debug(f"Entities: {interaction.entities}")

            return enhanced_prompt, has_intent
        except Exception as e:
            self.logger.error(f"Error building context: {e}")
            return "", False

