"""
Enhanced Context Processor for Mira with advanced NLP features.

This module provides a comprehensive context processor that includes:
- Advanced speaker recognition with clustering
- Database integration with Person entities
- NLP features (NER, coreference resolution, topic modeling)
- Improved conversation management
- Context summarization
"""

from __future__ import annotations

import re
import time
import json
import logging
from collections import defaultdict, deque
from datetime import datetime, timezone
from typing import Dict, List, Optional, Tuple, Any, Set
from dataclasses import dataclass, asdict, field
import uuid
import numpy as np
from sklearn.cluster import DBSCAN
from sqlalchemy.orm import Session
import spacy
from transformers.pipelines import pipeline
from sentence_transformers import SentenceTransformer
from models import Person, Interaction as DBInteraction, Conversation as DBConversation
from context_config import ContextProcessorConfig, DEFAULT_CONFIG
from db import get_db_session


@dataclass
class Interaction:
    """Enhanced interaction with additional NLP features."""
    timestamp: float
    formatted_timestamp: str
    speaker: str
    text: str
    raw_input: str
    person_id: Optional[str] = None
    entities: Optional[List[Dict]] = None
    topics: Optional[List[str]] = None
    sentiment: Optional[float] = None
    embedding: Optional[np.ndarray] = None

    def to_dict(self) -> Dict[str, Any]:
        data = asdict(self)
        if self.embedding is not None:
            data['embedding'] = self.embedding.tolist()
        return data


@dataclass
class SpeakerProfile:
    """Enhanced speaker profile with clustering information."""
    person_id: str
    speaker_index: int
    name: Optional[str] = None
    voice_embeddings: List[np.ndarray] = field(default_factory=list)
    cluster_id: Optional[int] = None
    confidence: float = 0.0
    interaction_count: int = 0
    is_identified: bool = False


class EnhancedContextProcessor:
    """
    Enhanced context processor with advanced NLP and speaker recognition features.
    """

    def __init__(self, config: Optional[ContextProcessorConfig] = None):
        """Initialize the enhanced context processor."""
        self.config = config or DEFAULT_CONFIG
        self.interaction_history: List[Interaction] = []
        self.speaker_profiles: Dict[int, SpeakerProfile] = {}
        self.keyword_index: Dict[str, List[int]] = defaultdict(list)
        self.conversation_cache: deque = deque(maxlen=100)
                
        # Clustering components
        self.dbscan = DBSCAN(
            eps=self.config.dbscan_eps,
            min_samples=self.config.dbscan_min_samples
        )
        self.voice_embedding_cache: Dict[str, np.ndarray] = {}
        
        # Conversation tracking
        self.current_conversation_id: Optional[uuid.UUID] = None
        self.current_participants: Set[str] = set()
        
        # Setup logging
        logging.basicConfig(level=getattr(logging, self.config.log_level))
        self.logger = logging.getLogger(__name__)
        
        # Initialize NLP components
        self._init_nlp_components()

    def _init_nlp_components(self):
        """Initialize NLP components based on configuration."""
        self.nlp_components = {}
            
        try:
            if self.config.enable_ner or self.config.enable_coreference:
                self.nlp_components['spacy'] = spacy.load(self.config.spacy_model)
                
            if self.config.enable_sentiment_analysis:
                self.nlp_components['sentiment'] = pipeline(
                    task="text-classification",
                    model="cardiffnlp/twitter-roberta-base-sentiment-latest",
                    top_k=None
                )
                
            if self.config.enable_topic_modeling:
                self.nlp_components['sentence_transformer'] = SentenceTransformer(
                    'all-MiniLM-L6-v2'
                )
                
        except Exception as e:
            self.logger.error(f"Failed to initialize NLP components: {e}")
            self.nlp_components = {}

    def parse_whisper_output(self, whisper_output: str) -> Optional[Interaction]:
        """Parse whisper output with enhanced processing."""
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
            timestamp = dt.timestamp()
        except ValueError:
            timestamp = time.time()

        interaction = Interaction(
            timestamp=timestamp,
            formatted_timestamp=formatted_timestamp,
            speaker=speaker,
            text=text.strip(),
            raw_input=whisper_output.strip(),
        )
        
        # Add NLP processing
        self._process_nlp_features(interaction)
        
        return interaction

    def _process_nlp_features(self, interaction: Interaction):
        """Process NLP features for an interaction."""
            
        try:
            # Named Entity Recognition
            if self.config.enable_ner and 'spacy' in self.nlp_components:
                doc = self.nlp_components['spacy'](interaction.text)
                interaction.entities = [
                    {"text": ent.text, "label": ent.label_, "start": ent.start_char, "end": ent.end_char}
                    for ent in doc.ents
                ]
                
            # Sentiment Analysis
            if self.config.enable_sentiment_analysis and 'sentiment' in self.nlp_components:
                sentiment_result = self.nlp_components['sentiment'](interaction.text)
                if sentiment_result and len(sentiment_result[0]) > 0:
                    # Get the positive sentiment score
                    positive_score = next(
                        (item['score'] for item in sentiment_result[0] if item['label'] == 'LABEL_2'), 
                        0.5
                    )
                    interaction.sentiment = positive_score
                    
            # Text Embedding for semantic similarity
            if self.config.enable_context_summarization and 'sentence_transformer' in self.nlp_components:
                interaction.embedding = self.nlp_components['sentence_transformer'].encode(
                    interaction.text
                )
                
        except Exception as e:
            self.logger.error(f"NLP processing failed: {e}")

    def get_or_create_person(self, speaker_index: int, session: Session) -> Person:
        """Get or create a person entity in the database."""
        person = session.query(Person).filter_by(speaker_index=speaker_index).first()
        
        if not person:
            person = Person(
                speaker_index=speaker_index,
                name=f"Person {speaker_index}",
                is_identified=False
            )
            session.add(person)
            session.commit()
            session.refresh(person)
            
        return person

    def update_speaker_clustering(self, voice_embedding: np.ndarray, speaker_index: int):
        """Update speaker clustering with new voice embedding."""
        if speaker_index not in self.speaker_profiles:
            self.speaker_profiles[speaker_index] = SpeakerProfile(
                person_id="",  # Will be set from database
                speaker_index=speaker_index
            )
            
        profile = self.speaker_profiles[speaker_index]
        profile.voice_embeddings.append(voice_embedding)
        profile.interaction_count += 1
        
        # Limit embedding history
        if len(profile.voice_embeddings) > 50:
            profile.voice_embeddings = profile.voice_embeddings[-25:]
            
        # Update clustering if we have enough data
        if len(profile.voice_embeddings) >= self.config.dbscan_min_samples:
            self._update_clusters()

    def _update_clusters(self):
        """Update DBSCAN clustering for all speakers."""
        all_embeddings = []
        embedding_to_speaker = []
        
        for speaker_idx, profile in self.speaker_profiles.items():
            for embedding in profile.voice_embeddings:
                all_embeddings.append(embedding)
                embedding_to_speaker.append(speaker_idx)
                
        if len(all_embeddings) < self.config.dbscan_min_samples:
            return
            
        try:
            all_embeddings = np.array(all_embeddings)
            cluster_labels = self.dbscan.fit_predict(all_embeddings)
            
            # Update speaker profiles with cluster information
            speaker_clusters = defaultdict(list)
            for i, speaker_idx in enumerate(embedding_to_speaker):
                speaker_clusters[speaker_idx].append(cluster_labels[i])
                
            for speaker_idx, clusters in speaker_clusters.items():
                # Use most common cluster as the speaker's cluster
                most_common_cluster = max(set(clusters), key=clusters.count)
                self.speaker_profiles[speaker_idx].cluster_id = most_common_cluster
                
        except Exception as e:
            self.logger.error(f"Clustering update failed: {e}")

    def add_interaction(self, interaction: Interaction, voice_embedding: Optional[np.ndarray] = None) -> None:
        """Add interaction with enhanced database integration."""
        self.interaction_history.append(interaction)
        
        # Extract speaker index
        speaker_match = re.search(r"Person (\d+)", interaction.speaker)
        speaker_index = int(speaker_match.group(1)) if speaker_match else 1
        
        # Update speaker clustering if embedding provided
        if voice_embedding is not None:
            self.update_speaker_clustering(voice_embedding, speaker_index)
        
        # Database integration
        try:
            session = get_db_session()
            
            # Get or create person
            person = self.get_or_create_person(speaker_index, session)
            interaction.person_id = str(person.id)
            
            # Create database interaction
            db_interaction = DBInteraction(
                user_id=speaker_index,  # Use user_id for backward compatibility
                speaker_id=person.id,
                text=interaction.text,
                timestamp=datetime.fromtimestamp(interaction.timestamp, tz=timezone.utc),
                entities=interaction.entities,
                topics=interaction.topics,
                sentiment=interaction.sentiment,
                conversation_id=self.current_conversation_id if self.current_conversation_id else None
            )
            
                
            session.add(db_interaction)
            session.commit()
            session.close()
            
        except Exception as e:
            self.logger.error(f"Database integration failed: {e}")
        
        # Update keyword index
        self._update_keyword_index(interaction)
        
        # Maintain history size limit
        if len(self.interaction_history) > self.config.max_history_size:
            self._cleanup_old_interactions()

    def _update_keyword_index(self, interaction: Interaction):
        """Update keyword index with enhanced text processing."""
        words = interaction.text.lower().split()
        interaction_index = len(self.interaction_history) - 1

        for word in words:
            clean_word = re.sub(r"[^\w]", "", word)
            if clean_word and len(clean_word) > 2:
                self.keyword_index[clean_word].append(interaction_index)
                
        # Add entities as keywords
        if interaction.entities:
            for entity in interaction.entities:
                entity_text = entity['text'].lower().replace(' ', '_')
                self.keyword_index[entity_text].append(interaction_index)

    def detect_conversation_boundary(self, current_interaction: Interaction) -> bool:
        """Enhanced conversation boundary detection."""
        if not self.interaction_history:
            return True
            
        last_interaction = self.interaction_history[-1]
        
        # Time gap detection
        time_gap = current_interaction.timestamp - last_interaction.timestamp
        if time_gap > self.config.conversation_gap_threshold:
            return True
            
        # Speaker change with extended silence
        if (current_interaction.speaker != last_interaction.speaker and 
            time_gap > self.config.conversation_gap_threshold / 2):
            return True
            
        # Topic shift detection using embeddings
        if (self.config.enable_topic_modeling and 
            current_interaction.embedding is not None and 
            last_interaction.embedding is not None):
            
            similarity = np.dot(current_interaction.embedding, last_interaction.embedding) / (
                np.linalg.norm(current_interaction.embedding) * 
                np.linalg.norm(last_interaction.embedding)
            )
            
            if similarity < self.config.context_similarity_threshold:
                return True
                
        return False

    def get_short_term_context(self, current_time: float) -> List[Interaction]:
        """Enhanced short-term context retrieval."""
        if not self.interaction_history:
            return []

        # Find current interaction index
        current_index = -1
        for i in range(len(self.interaction_history) - 1, -1, -1):
            if self.interaction_history[i].timestamp <= current_time:
                current_index = i
                break

        if current_index == -1:
            return []

        # Find conversation start with enhanced detection
        conversation_start = self._find_conversation_start_enhanced(current_index)
        
        # Get conversation interactions
        conversation_interactions = self.interaction_history[
            conversation_start : current_index + 1
        ]

        # Limit to max conversation length
        if len(conversation_interactions) > self.config.max_conversation_length:
            conversation_interactions = conversation_interactions[
                -self.config.max_conversation_length :
            ]

        return conversation_interactions

    def _find_conversation_start_enhanced(self, current_index: int) -> int:
        """Enhanced conversation start detection."""
        if current_index < 0 or current_index >= len(self.interaction_history):
            return max(0, current_index)

        speakers = set()
        conversation_start = current_index
        topics = []

        for i in range(current_index, -1, -1):
            interaction = self.interaction_history[i]
            speakers.add(interaction.speaker)
            
            # Speaker limit check
            if len(speakers) > 2:
                conversation_start = i + 1
                break
                
            # Time gap check
            if i > 0:
                prev_interaction = self.interaction_history[i - 1]
                time_gap = interaction.timestamp - prev_interaction.timestamp
                if time_gap > self.config.conversation_gap_threshold:
                    conversation_start = i
                    break
                    
            # Topic coherence check
            if (self.config.enable_topic_modeling and 
                interaction.embedding is not None):
                topics.append(interaction.embedding)
                
                if len(topics) > 5:  # Check topic coherence every 5 interactions
                    avg_similarity = self._calculate_topic_coherence(topics)
                    if avg_similarity < self.config.context_similarity_threshold:
                        conversation_start = i + 2  # Start after the topic shift
                        break

            conversation_start = i

        return conversation_start

    def _calculate_topic_coherence(self, embeddings: List[np.ndarray]) -> float:
        """Calculate average topic coherence from embeddings."""
        if len(embeddings) < 2:
            return 1.0
            
        similarities = []
        for i in range(len(embeddings) - 1):
            for j in range(i + 1, len(embeddings)):
                sim = np.dot(embeddings[i], embeddings[j]) / (
                    np.linalg.norm(embeddings[i]) * np.linalg.norm(embeddings[j])
                )
                similarities.append(sim)
                
        return float(np.mean(similarities)) if similarities else 1.0

    def get_long_term_context(
        self, keywords: List[str], current_interaction: Optional[Interaction] = None, 
        max_results: Optional[int] = None
    ) -> List[Interaction]:
        """Enhanced long-term context retrieval with semantic similarity."""
        max_results = max_results or self.config.long_term_context_max_results
        
        if not keywords and current_interaction is None:
            return []
            
        relevant_interactions = []
        
        # Keyword-based retrieval (existing functionality)
        if keywords:
            interaction_scores = self._score_interactions_by_keywords(keywords)
            keyword_results = self._get_top_scored_interactions(interaction_scores, max_results)
            relevant_interactions.extend(keyword_results)
        
        # Semantic similarity retrieval
        if (current_interaction and current_interaction.embedding is not None and 
            self.config.enable_context_summarization):
            semantic_results = self._get_semantic_similar_interactions(
                current_interaction.embedding, max_results
            )
            relevant_interactions.extend(semantic_results)
        
        # Remove duplicates and sort by relevance
        seen_indices = set()
        unique_interactions = []
        for interaction in relevant_interactions:
            interaction_index = self.interaction_history.index(interaction)
            if interaction_index not in seen_indices:
                seen_indices.add(interaction_index)
                unique_interactions.append(interaction)
                
        return unique_interactions[:max_results]

    def _score_interactions_by_keywords(self, keywords: List[str]) -> Dict[int, float]:
        """Score interactions by keyword matches with entity weighting."""
        interaction_scores = defaultdict(float)

        for keyword in keywords:
            clean_keyword = re.sub(r"[^\w]", "", keyword.lower())
            if clean_keyword in self.keyword_index:
                for interaction_idx in self.keyword_index[clean_keyword]:
                    interaction = self.interaction_history[interaction_idx]
                    
                    # Base keyword score
                    score = 1.0
                    
                    # Boost score if keyword matches an entity
                    if interaction.entities:
                        for entity in interaction.entities:
                            if clean_keyword in entity['text'].lower():
                                score *= 2.0  # Entity match bonus
                                
                    interaction_scores[interaction_idx] += score

        return interaction_scores

    def _get_semantic_similar_interactions(
        self, query_embedding: np.ndarray, max_results: int
    ) -> List[Interaction]:
        """Get semantically similar interactions using embeddings."""
        similarities = []
        
        for i, interaction in enumerate(self.interaction_history):
            if interaction.embedding is not None:
                similarity = np.dot(query_embedding, interaction.embedding) / (
                    np.linalg.norm(query_embedding) * np.linalg.norm(interaction.embedding)
                )
                if similarity >= self.config.context_similarity_threshold:
                    similarities.append((i, similarity))
        
        # Sort by similarity and return top results
        similarities.sort(key=lambda x: x[1], reverse=True)
        return [
            self.interaction_history[idx] 
            for idx, _ in similarities[:max_results]
        ]

    def _get_top_scored_interactions(
        self, interaction_scores: Dict[int, float], max_results: int
    ) -> List[Interaction]:
        """Get top scored interactions sorted by score and recency."""
        sorted_interactions = sorted(
            interaction_scores.items(),
            key=lambda x: (x[1], self.interaction_history[x[0]].timestamp),
            reverse=True,
        )
        
        result_indices = [idx for idx, _ in sorted_interactions[:max_results]]
        return [self.interaction_history[idx] for idx in result_indices]

    def build_context_prompt(self, current_text: str, current_time: float) -> str:
        """Build enhanced context prompt with summarization."""
        # Get contexts
        short_term = self.get_short_term_context(current_time)
        if short_term and current_text in short_term[-1].text:
            short_term = short_term[:-1]

        keywords = self.extract_keywords(current_text)
        
        # For long-term context, try to get current interaction for semantic similarity
        current_interaction = None
        if self.interaction_history:
            current_interaction = self.interaction_history[-1]
            
        long_term = self.get_long_term_context(keywords, current_interaction)
        long_term = [i for i in long_term if i.text != current_text]

        # Build context with summarization
        context_parts = []

        if short_term:
            context_parts.append("Current conversation:")
            for interaction in short_term:
                speaker_info = interaction.speaker
                if interaction.entities:
                    entity_text = ", ".join([e['text'] for e in interaction.entities[:3]])
                    speaker_info += f" [entities: {entity_text}]"
                    
                context_parts.append(f"\n- {speaker_info}: {interaction.text}")

        if long_term:
            if self.config.enable_context_summarization and len(long_term) > 3:
                # Summarize long-term context
                summary = self._summarize_context(long_term)
                context_parts.append(f"\n\nRelevant context summary: {summary}")
            else:
                context_parts.append("\n\nRelevant historical context:")
                for interaction in long_term:
                    context_parts.append(
                        f"\n- {interaction.formatted_timestamp} {interaction.speaker}: {interaction.text}"
                    )

        return "".join(context_parts) if context_parts else ""

    def _summarize_context(self, interactions: List[Interaction]) -> str:
        """Summarize context interactions."""
        if not interactions:
            return ""
            
        # Simple extractive summarization
        texts = [i.text for i in interactions]
        
        # Combine entities and key phrases
        key_info = []
        for interaction in interactions:
            if interaction.entities:
                entities = [e['text'] for e in interaction.entities if e['label'] in ['PERSON', 'ORG', 'EVENT']]
                key_info.extend(entities)
        
        # Create summary
        combined_text = " ".join(texts)
        sentences = combined_text.split(". ")
        
        # Keep the most informative sentences (simple heuristic)
        important_sentences = []
        for sentence in sentences[:5]:  # Limit to first 5 sentences
            if any(keyword in sentence.lower() for keyword in key_info[:10]):
                important_sentences.append(sentence)
                
        summary = ". ".join(important_sentences[:3])  # Max 3 sentences
        
        if len(summary) > self.config.summary_max_length:
            summary = summary[:self.config.summary_max_length] + "..."
            
        return summary or "Previous discussion about relevant topics."

    def extract_keywords(self, text: str) -> List[str]:
        """Enhanced keyword extraction with NLP features."""
        # Base keyword extraction (existing logic)
        words = text.lower().split()
        stop_words = {
            "the", "a", "an", "and", "or", "but", "in", "on", "at", "to", "for", "of",
            "with", "by", "is", "am", "are", "was", "were", "be", "been", "being",
            "have", "has", "had", "do", "does", "did", "will", "would", "could",
            "should", "may", "might", "can", "i", "me", "my", "myself", "you", "your",
            "yours", "yourself", "he", "him", "his", "himself", "she", "her", "hers",
            "herself", "it", "its", "itself", "we", "us", "our", "ours", "ourselves",
            "they", "them", "their", "theirs", "themselves",
        }

        keywords = []
        for word in words:
            clean_word = re.sub(r"[^\w]", "", word)
            if clean_word and clean_word not in stop_words and len(clean_word) > 2:
                keywords.append(clean_word)

        return list(set(keywords))  # Remove duplicates

    def classify_intent(self, text: str) -> bool:
        """Enhanced intent classification with NLP features."""
        # Base classification (existing logic)
        action_keywords = {
            "contact": ["call", "text", "message", "email", "tell", "contact", "reach out"],
            "remind": ["remind", "remember", "later", "tomorrow", "next week", "schedule reminder"],
            "schedule": ["schedule", "appointment", "meeting", "book", "plan", "event", "calendar",
                        "let's", "let us", "we'll", "we will", "we're", "we are", "we can",
                        "we could", "we should", "we might"],
            "generic": ["yes", "okay", "sure", "alright", "sounds good", "go ahead", "do it",
                       "i'm", "i'll", "bet"],
        }

        text_lower = text.lower()
        has_keywords = any(
            any(keyword in text_lower for keyword in keywords)
            for keywords in action_keywords.values()
        )

        if has_keywords:
            return True

        return False

    def process_input(self, whisper_output: str, voice_embedding: Optional[np.ndarray] = None) -> Tuple[str, bool]:
        """Enhanced input processing with full feature integration."""
        # Parse whisper output
        interaction = self.parse_whisper_output(whisper_output)
        if not interaction:
            return whisper_output, False

        # Check for conversation boundary
        if self.detect_conversation_boundary(interaction):
            self._start_new_conversation(interaction)

        # Add to history with database integration
        self.add_interaction(interaction, voice_embedding)

        # Build enhanced context
        enhanced_prompt = self.build_context_prompt(
            interaction.text, interaction.timestamp
        )

        # Enhanced intent classification
        has_intent = self.classify_intent(interaction.text)

        if self.config.debug_mode:
            self.logger.debug(f"Processed interaction: {interaction.text}")
            self.logger.debug(f"Intent detected: {has_intent}")
            self.logger.debug(f"Entities: {interaction.entities}")

        return enhanced_prompt, has_intent

    def _start_new_conversation(self, interaction: Interaction):
        """Start a new conversation in the database."""
        try:
            session = get_db_session()
            
            # Create new conversation
            conversation = DBConversation(
                user_ids=str(interaction.person_id) if interaction.person_id else "unknown",  # Provide default value
                speaker_id=interaction.person_id,
                start_of_conversation=datetime.fromtimestamp(interaction.timestamp, tz=timezone.utc),
                participants=json.dumps([interaction.person_id] if interaction.person_id else [])
            )
            
            session.add(conversation)
            session.commit()
            session.refresh(conversation)
            
            self.current_conversation_id = getattr(conversation, "id", None)
            self.current_participants = {interaction.person_id} if interaction.person_id else set()
            
            session.close()
            
        except Exception as e:
            self.logger.error(f"Failed to start new conversation: {e}")

    def _cleanup_old_interactions(self):
        """Enhanced cleanup with preserved relationships."""
        excess_count = len(self.interaction_history) - self.config.max_history_size
        if excess_count > 0:
            # Remove oldest interactions
            removed_interactions = self.interaction_history[:excess_count]
            self.interaction_history = self.interaction_history[excess_count:]

            # Rebuild keyword index efficiently
            self.keyword_index.clear()
            for i, interaction in enumerate(self.interaction_history):
                self._update_keyword_index_for_cleanup(interaction, i)

    def _update_keyword_index_for_cleanup(self, interaction: Interaction, new_index: int):
        """Update keyword index during cleanup."""
        words = interaction.text.lower().split()
        for word in words:
            clean_word = re.sub(r"[^\w]", "", word)
            if clean_word and len(clean_word) > 2:
                self.keyword_index[clean_word].append(new_index)

    def get_speaker_summary(self) -> Dict[str, Any]:
        """Get summary of all tracked speakers."""
        summary = {}
        for speaker_idx, profile in self.speaker_profiles.items():
            summary[f"Person {speaker_idx}"] = {
                "interaction_count": profile.interaction_count,
                "cluster_id": profile.cluster_id,
                "is_identified": profile.is_identified,
                "name": profile.name
            }
        return summary


# Utility functions for integration
def create_enhanced_context_processor(config: Optional[ContextProcessorConfig] = None) -> EnhancedContextProcessor:
    """Create a new enhanced context processor instance."""
    return EnhancedContextProcessor(config)


def process_interaction_enhanced(
    processor: EnhancedContextProcessor, 
    interaction: DBInteraction,
    voice_embedding: Optional[np.ndarray] = None
) -> Tuple[str, bool]:
    """Process a database interaction with enhanced features."""
    # Format the interaction as the processor expects
    timestamp_str = interaction.timestamp.strftime("%Y-%m-%d %H:%M:%S")
    formatted_input = f"({timestamp_str}) Person {interaction.speaker}: {interaction.text}"
    return processor.process_input(formatted_input, voice_embedding)