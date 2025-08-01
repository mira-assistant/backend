from __future__ import annotations

import re
import json
import logging
from collections import deque
from datetime import datetime, timezone, timedelta
from typing import Dict, List, Optional, Tuple
import uuid
import numpy as np

from sklearn.cluster import DBSCAN
import spacy
from transformers.pipelines import pipeline
from sentence_transformers import SentenceTransformer

from sqlalchemy.orm import Session
from sqlalchemy import or_

from models import (
    Person,
    Interaction,
    Conversation,
)
from db import get_db_session

from typing import Literal


class ContextProcessorConfig:
    """Configuration class for the context processor."""

    class SpeakerRecognitionConfig:
        """Speaker recognition parameters."""

        SPEAKER_SIMILARITY_THRESHOLD: float = 0.7
        """Cosine similarity threshold for determining if two voice samples are from the same speaker.
        Lowering this value decreases sensitivity, making it less likely to group different speakers together,
        but may increase false negatives (splitting the same speaker into multiple clusters)."""
        DBSCAN_EPS: float = 0.9
        """Epsilon parameter for DBSCAN clustering algorithm, controlling the maximum distance between samples in a cluster."""
        DBSCAN_MIN_SAMPLES: Literal[2] = 2
        """Minimum number of samples required to form a cluster in DBSCAN."""

    class NLPConfig:
        """Natural Language Processing parameters."""

        SPACY_MODEL: Literal["en_core_web_sm"] = "en_core_web_sm"
        """spaCy language model used for natural language processing tasks."""
        CONTEXT_SIMILARITY_THRESHOLD: float = 0.7
        """Threshold for semantic similarity when comparing contexts.
        Lowering this value makes the system more likely to consider contexts as similar,
        potentially increasing recall but reducing precision."""

    class ContextManagementParameters:
        """Parameters for managing context and conversation boundaries."""

        CONVERSATION_GAP_THRESHOLD: Literal[300] = 300
        """Time gap in seconds used to determine conversation boundaries.
        Lowering this value will result in more frequent splitting of conversations."""
        SHORT_TERM_CONTEXT_MAX_RESULTS: Literal[20] = 20
        """Maximum number of recent interactions to include in the short-term context."""
        LONG_TERM_CONTEXT_MAX_RESULTS: Literal[5] = 5
        """Maximum number of results to retrieve from long-term context storage."""
        SUMMARY_MAX_LENGTH: Literal[200] = 200
        """Maximum length (in tokens or characters) for generated context summaries."""

    class PerformanceConfig:
        """Performance optimization parameters."""

        BATCH_PROCESSING: Literal[False] = False
        """Enable or disable batch processing to improve performance on large datasets."""
        CACHE_EMBEDDINGS: Literal[True] = True
        """Enable or disable caching of voice and text embeddings to speed up repeated computations."""
        ASYNC_PROCESSING: Literal[False] = False
        """Enable or disable asynchronous processing for improved throughput."""

    class DebugConfig:
        """Debugging and logging parameters."""

        DEBUG_MODE: Literal[False] = False
        """Enable or disable debug mode for verbose logging and additional diagnostics."""
        LOG_LEVEL: Literal["INFO"] = "INFO"
        """Logging level for controlling the verbosity of log output."""


class ContextProcessor:
    """
    Context processor with advanced NLP and speaker recognition features.
    """

    def __init__(self):
        """Initialize the context processor."""
        self.conversation_cache: deque = deque(maxlen=100)

        # Speaker detection state variables for advanced clustering
        self._speaker_embeddings: List[np.ndarray] = []
        self._speaker_ids: List[uuid.UUID] = []
        self._speaker_interaction_ids: List[uuid.UUID] = []
        self._cluster_labels: List[int] = []
        self._clusters_dirty: bool = True

        self.dbscan = DBSCAN(
            eps=ContextProcessorConfig.SpeakerRecognitionConfig.DBSCAN_EPS,
            min_samples=ContextProcessorConfig.SpeakerRecognitionConfig.DBSCAN_MIN_SAMPLES,
            metric="cosine",
        )

        self.current_conversation_id = None
        self.current_participants = set()

        logging.basicConfig(level=getattr(logging, ContextProcessorConfig.DebugConfig.LOG_LEVEL))
        self.logger = logging.getLogger(__name__)

        self._init_nlp_components()

    def _init_nlp_components(self):
        self.nlp_components = {}

        self.nlp_components["spacy"] = spacy.load(ContextProcessorConfig.NLPConfig.SPACY_MODEL)
        self.nlp_components["sentence_transformer"] = SentenceTransformer("all-MiniLM-L6-v2")
        self.nlp_components["sentiment"] = pipeline(
            task="text-classification",
            model="cardiffnlp/twitter-roberta-base-sentiment-latest",
            top_k=None,
        )

    # --- ADVANCED SPEAKER DETECTION SECTION START ---

    def _refresh_speaker_cache(self):
        """(Re)load all speaker embeddings, person_ids, interaction_ids from the database."""
        session = get_db_session()
        try:
            interactions = (
                session.query(Interaction)
                .filter(Interaction.voice_embedding.isnot(None), Interaction.speaker_id.isnot(None))
                .all()
            )
            self._speaker_embeddings = [
                np.array(i.voice_embedding, dtype=np.float32) for i in interactions
            ]
            self._speaker_ids = [i.speaker_id for i in interactions]
            self._speaker_interaction_ids = [i.id for i in interactions]
            self._clusters_dirty = True
        finally:
            session.close()

    def _recompute_clusters(self):
        """Run DBSCAN clustering on all cached embeddings and update labels."""
        if not self._speaker_embeddings or (
            isinstance(self._speaker_embeddings, np.ndarray) and self._speaker_embeddings.size == 0
        ):
            self._cluster_labels = []
            return
        X = np.stack(self._speaker_embeddings)
        self._cluster_labels = self.dbscan.fit_predict(X).tolist()
        self._clusters_dirty = False

    def assign_speaker(
        self,
        embedding: np.ndarray,
        session: Optional[Session] = None,
        interaction_id=None,
    ):
        """
        Assign a speaker using DBSCAN clustering over all embeddings.
        Returns the Person.id of the most similar speaker (if above threshold) or creates a new one.
        Also updates clusters in the database.

        Args:
            embedding: The new voice embedding (np.ndarray)
            session: Optional SQLAlchemy session to reuse
            interaction_id: The interaction UUID to use for the new embedding, if available
        """

        embedding = np.array(embedding, dtype=np.float32)

        own_session = session is None
        session = session or get_db_session()
        try:
            if (
                not self._speaker_embeddings
                or (
                    isinstance(self._speaker_embeddings, np.ndarray)
                    and self._speaker_embeddings.size == 0
                )
                or self._clusters_dirty
            ):
                self._refresh_speaker_cache()
                self._recompute_clusters()
                self._recompute_clusters()

            # Add the new embedding to the cached ones for clustering
            all_embeddings = self._speaker_embeddings + [embedding]
            X = np.stack(all_embeddings)
            dbscan = DBSCAN(
                eps=ContextProcessorConfig.SpeakerRecognitionConfig.DBSCAN_EPS,
                min_samples=ContextProcessorConfig.SpeakerRecognitionConfig.DBSCAN_MIN_SAMPLES,
                metric="cosine",
            )

            labels = dbscan.fit_predict(X)
            new_label = labels[-1]

            # Helper to append to cache with correct types
            def _append_cache(embedding, person_id, interaction_id):
                self._speaker_embeddings.append(embedding)
                self._speaker_ids.append(person_id)
                self._speaker_interaction_ids.append(interaction_id)
                self._clusters_dirty = True

            if new_label == -1:
                new_index = (
                    session.query(Person.index).order_by(Person.index.desc()).first() or [0]
                )[0] + 1
                new_person = Person(
                    voice_embedding=(
                        embedding.tolist() if isinstance(embedding, np.ndarray) else embedding
                    ),
                    index=new_index,
                )
                session.add(new_person)
                session.commit()
                session.refresh(new_person)
                _append_cache(embedding, new_person.id, interaction_id)
                self._update_db_clusters(session, labels[:-1] + [-1])
                return new_person.id

            cluster_indices = [i for i, label in enumerate(labels[:-1]) if label == new_label]
            if not cluster_indices:
                new_index = (
                    session.query(Person.index).order_by(Person.index.desc()).first() or [0]
                )[0] + 1
                new_person = Person(
                    voice_embedding=embedding.tolist(),
                    index=new_index,
                )
                session.add(new_person)
                session.commit()
                session.refresh(new_person)
                _append_cache(embedding, new_person.id, interaction_id)
                self._update_db_clusters(session, labels[:-1] + [-1])
                return new_person.id

            similarities = []
            for idx in cluster_indices:
                emb = all_embeddings[idx]
                sim = float(
                    np.dot(embedding, emb) / (np.linalg.norm(embedding) * np.linalg.norm(emb))
                )
                similarities.append((idx, sim))
            best_idx, best_sim = max(similarities, key=lambda x: x[1])

            print("best similarity:", best_sim)

            if (
                best_sim
                < ContextProcessorConfig.SpeakerRecognitionConfig.SPEAKER_SIMILARITY_THRESHOLD
            ):
                new_index = (
                    session.query(Person.index).order_by(Person.index.desc()).first() or [0]
                )[0] + 1
                new_person = Person(
                    voice_embedding=embedding.tolist(),
                    index=new_index,
                )
                session.add(new_person)
                session.commit()
                session.refresh(new_person)
                _append_cache(embedding, new_person.id, interaction_id)
                self._update_db_clusters(session, labels[:-1] + [-1])
                return new_person.id

            # Assign to the Person of the best match in the cluster
            matched_person_id = self._speaker_ids[best_idx]
            matched_person = session.query(Person).filter_by(id=matched_person_id).first()
            if matched_person and matched_person.voice_embedding is not None:
                old_emb = np.array(matched_person.voice_embedding, dtype=np.float32)
                updated_emb = 0.8 * old_emb + 0.2 * embedding
                matched_person.voice_embedding = updated_emb.tolist()
                session.commit()
            _append_cache(embedding, matched_person_id, interaction_id)
            self._update_db_clusters(session, labels)
            return matched_person_id
        finally:
            if own_session:
                session.close()

    def _update_db_clusters(self, session: Session, cluster_labels):
        """
        Update DB cluster assignments for all Persons based on current cache and cluster_labels.
        Interactions in cache must match the order of cluster_labels.
        """
        if (
            len(self._speaker_ids) == 0
            or len(cluster_labels) == 0
            or len(self._speaker_ids) != len(cluster_labels)
        ):
            return
        for person_id, label in zip(self._speaker_ids, cluster_labels):
            if person_id is None:
                continue
            person = session.query(Person).filter_by(id=person_id).first()
            if person:
                person.cluster_id = int(label) if label != -1 else None
        session.commit()

    # --- ADVANCED SPEAKER DETECTION SECTION END ---

    # ... all other ContextProcessor methods remain unchanged ...

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
            doc = self.nlp_components["spacy"](interaction.text)
            entities_list = [
                {
                    "text": ent.text,
                    "label": ent.label_,
                    "start": ent.start_char,
                    "end": ent.end_char,
                }
                for ent in doc.ents
            ]
            # Store as JSON if the column is JSON type, else as string
            try:
                interaction.entities = entities_list
            except Exception:
                interaction.entities = json.dumps(entities_list)

            # Sentiment Analysis
            sentiment_result = self.nlp_components["sentiment"](interaction.text)
            if sentiment_result and len(sentiment_result[0]) > 0:
                # Get the positive sentiment score
                positive_score = next(
                    (item["score"] for item in sentiment_result[0] if item["label"] == "LABEL_2"),
                    0.5,
                )
                interaction.sentiment = positive_score

            # Text Embedding for semantic similarity
            embedding = self.nlp_components["sentence_transformer"].encode(interaction.text)
            # Store embedding as JSON in voice_embedding field (reusing existing field)
            interaction.voice_embedding = embedding.tolist()

        except Exception as e:
            self.logger.error(f"NLP processing failed: {e}")

    def get_or_create_person(self, speaker_index: int, session: Session) -> Person:
        """Get or create a person entity in the database."""
        person = session.query(Person).filter_by(index=speaker_index).first()

        if not person:
            person = Person(
                index=speaker_index,
            )
            session.add(person)
            session.commit()
            session.refresh(person)

        return person

    def _cosine_sim(self, a: np.ndarray, b: np.ndarray) -> float:
        """Cosine similarity between two vectors."""
        return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))

    def _update_clusters_db(self, session):
        """Update DBSCAN clustering for all speakers using database."""
        persons = session.query(Person).filter(Person.voice_embedding.isnot(None)).all()

        if len(persons) < ContextProcessorConfig.SpeakerRecognitionConfig.DBSCAN_MIN_SAMPLES:
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
                interaction.speaker_id = person_id

            # Check if we need to start a new conversation or continue existing one
            if self.detect_conversation_boundary(interaction):
                self._start_new_conversation(interaction, session)
            else:
                # Assign to current conversation if exists
                if self.current_conversation_id is not None:
                    interaction.conversation_id = self.current_conversation_id

            # Save interaction to database
            session.add(interaction)
            session.commit()
            session.refresh(interaction)

            # Update current conversation participants
            if interaction.speaker_id is not None:
                self.current_participants.add(interaction.speaker_id)

            self.logger.debug(f"Interaction added to database with ID: {interaction.id}")

        except Exception as e:
            session.rollback()
            self.logger.error(f"Database integration failed: {e}")
        finally:
            session.close()

    def detect_conversation_boundary(self, current_interaction: Interaction) -> bool:
        """Conversation boundary detection using database queries."""
        # Get last interaction from database
        session = get_db_session()

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
        if time_gap > ContextProcessorConfig.ContextManagementParameters.CONVERSATION_GAP_THRESHOLD:
            return True

        # Speaker change with extended silence
        if (
            current_interaction.speaker_id != last_interaction.speaker_id
            and time_gap
            > ContextProcessorConfig.ContextManagementParameters.CONVERSATION_GAP_THRESHOLD / 2
        ):
            return True

        # Topic shift detection using embeddings if available
        if (
            current_interaction.voice_embedding is not None
            and last_interaction.voice_embedding is not None
        ):
            current_embedding = np.array(current_interaction.voice_embedding)
            last_embedding = np.array(last_interaction.voice_embedding)

            similarity = self._cosine_sim(current_embedding, last_embedding)

            if (
                similarity
                < ContextProcessorConfig.ContextManagementParameters.context_similarity_threshold
            ):
                return True

        return False

    def get_short_term_context(self, current_time) -> List[Interaction]:
        """Short-term context retrieval from database."""
        session = get_db_session()
        try:
            # Get interactions from current conversation or recent interactions
            if self.current_conversation_id is not None:
                # Get interactions from current conversation
                interactions = (
                    session.query(Interaction)
                    .filter_by(conversation_id=self.current_conversation_id)
                    .order_by(Interaction.timestamp.desc())
                    .limit(
                        ContextProcessorConfig.ContextManagementParameters.SHORT_TERM_CONTEXT_MAX_RESULTS
                    )
                    .all()
                )
                interactions.reverse()  # Return in chronological order
            else:
                # Get recent interactions within time window
                time_threshold = current_time - timedelta(
                    seconds=ContextProcessorConfig.ContextManagementParameters.CONVERSATION_GAP_THRESHOLD
                )
                interactions = (
                    session.query(Interaction)
                    .filter(Interaction.timestamp >= time_threshold)
                    .filter(Interaction.timestamp <= current_time)
                    .order_by(Interaction.timestamp.asc())
                    .limit(
                        ContextProcessorConfig.ContextManagementParameters.SHORT_TERM_CONTEXT_MAX_RESULTS
                    )
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
        max_results = (
            max_results
            or ContextProcessorConfig.ContextManagementParameters.LONG_TERM_CONTEXT_MAX_RESULTS
        )
        session = get_db_session()

        try:
            relevant_interactions = []

            # Keyword-based retrieval
            if keywords:
                keyword_results = self._get_keyword_interactions_db(keywords, session, max_results)
                relevant_interactions.extend(keyword_results)

            # Semantic similarity retrieval using embeddings
            if current_interaction and current_interaction.voice_embedding is not None:
                semantic_results = self._get_semantic_similar_interactions_db(
                    current_interaction.voice_embedding, session, max_results
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
        self, query_embedding, session, max_results: int
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
                if (
                    similarity
                    >= ContextProcessorConfig.ContextManagementParameters.context_similarity_threshold
                ):
                    similarities.append((interaction, similarity))
            except Exception as e:
                self.logger.debug(f"Error computing similarity: {e}")

        # Sort by similarity and return top results
        similarities.sort(key=lambda x: x[1], reverse=True)
        return [interaction for interaction, _ in similarities[:max_results]]

    # Removed obsolete in-memory scoring methods (replaced by database queries)

    def build_context_prompt(self, interaction: Interaction) -> str:
        """Build context prompt with summarization using database."""
        # Get contexts from database
        short_term = self.get_short_term_context(interaction.timestamp)

        # Remove current interaction if it's already in short term context
        if short_term and interaction.text in short_term[-1].text:
            short_term = short_term[:-1]

        keywords = self.extract_keywords(interaction.text)

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
        long_term = [i for i in long_term if str(i.text) != interaction.text]

        # Build context with summarization
        context_parts = []

        if short_term:
            context_parts.append("Current conversation:")
            for interaction in short_term:
                speaker_info = f"Person {self._get_speaker_index(interaction.speaker_id)}"
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

    def _get_speaker_index(self, speaker_id):
        """Get speaker index from person ID."""
        if not speaker_id:
            return 1

        session = get_db_session()
        try:
            person = session.query(Person).filter_by(id=speaker_id).first()
            return person.index if person else 1
        finally:
            session.close()

    def _summarize_context_db(self, interactions: List[Interaction]) -> str:
        """Summarize context interactions from database models."""
        if not interactions:
            return ""

        # Simple extractive summarization
        texts = [i.text for i in interactions]

        # Combine entities and key phrases
        key_info = []
        for interaction in interactions:
            if interaction.entities is not None:
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

        if len(summary) > ContextProcessorConfig.ContextManagementParameters.SUMMARY_MAX_LENGTH:
            summary = (
                summary[: ContextProcessorConfig.ContextManagementParameters.SUMMARY_MAX_LENGTH]
                + "..."
            )

        return summary or "Previous discussion about relevant topics."

    def extract_keywords(self, text) -> List[str]:
        """Keyword extraction with NLP features."""
        # Base keyword extraction (existing logic)
        words = str(text).lower().split()
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

    def classify_intent(self, text) -> bool:
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

        text_lower = str(text).lower()
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
            if self.detect_conversation_boundary(interaction):
                self._start_new_conversation(interaction, session)

            # Add to database with database integration
            self.add_interaction(interaction, voice_embedding)

            # Build context using database
            enhanced_prompt = self.build_context_prompt(interaction)

            # Enhanced intent classification
            has_intent = self.classify_intent(interaction.text)

            if ContextProcessorConfig.DebugConfig.DEBUG_MODE:
                self.logger.debug(f"Processed interaction: {interaction.text}")
                self.logger.debug(f"Intent detected: {has_intent}")
                self.logger.debug(f"Entities: {interaction.entities}")

            return enhanced_prompt, has_intent

        finally:
            session.close()

    def _start_new_conversation(self, interaction: Interaction, session):
        """Start a new conversation in the database with real-time tracking."""
        try:
            # Create new conversation
            conversation = Conversation(
                user_ids=(
                    interaction.speaker_id if interaction.speaker_id is not None else uuid.uuid4()
                ),
                speaker_id=interaction.speaker_id,
                start_of_conversation=interaction.timestamp,
                participants=json.dumps(
                    [str(interaction.speaker_id)] if interaction.speaker_id is not None else []
                ),
            )

            session.add(conversation)
            session.commit()
            session.refresh(conversation)

            self.current_conversation_id = conversation.id
            self.current_participants = (
                {interaction.speaker_id} if interaction.speaker_id is not None else set()
            )

            # Assign this conversation to the interaction
            interaction.conversation_id = conversation.id

            self.logger.debug(f"Started new conversation with ID: {conversation.id}")

        except Exception as e:
            self.logger.error(f"Failed to start new conversation: {e}")


# Utility functions for integration
def create_context_processor() -> ContextProcessor:
    """Create a new context processor instance."""
    return ContextProcessor()


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
