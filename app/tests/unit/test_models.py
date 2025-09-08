"""
Unit tests for database models.
"""

import uuid
from datetime import datetime

import pytest
from sqlalchemy.exc import IntegrityError

from app.models import Action, Conversation, Interaction, MiraNetwork, Person
from app.models.conversation import person_conversation_association


class TestMiraNetwork:
    """Test cases for MiraNetwork model."""

    def test_create_network(self, db_session):
        """Test creating a MiraNetwork instance."""
        network = MiraNetwork(
            name="Test Network",
            service_enabled=True,
            network_settings={"test": "value"},
        )

        db_session.add(network)
        db_session.commit()
        db_session.refresh(network)

        assert network.id is not None
        assert network.name == "Test Network"  # type: ignore
        assert network.service_enabled is True
        assert network.network_settings == {"test": "value"}  # type: ignore
        assert network.created_at is not None
        assert network.updated_at is not None

    def test_network_relationships(self, db_session):
        """Test MiraNetwork relationships."""
        network = MiraNetwork(name="Test Network")
        db_session.add(network)
        db_session.commit()
        db_session.refresh(network)

        # Test that relationships are properly initialized
        assert network.interactions == []
        assert network.persons == []
        assert network.conversations == []
        assert network.actions == []

    def test_network_timestamps(self, db_session):
        """Test that timestamps are automatically set."""
        network = MiraNetwork(name="Test Network")
        db_session.add(network)
        db_session.commit()
        db_session.refresh(network)

        assert network.created_at is not None
        assert network.updated_at is not None
        assert isinstance(network.created_at, datetime)
        assert isinstance(network.updated_at, datetime)


class TestPerson:
    """Test cases for Person model."""

    def test_create_person(self, db_session, sample_network_id):
        """Test creating a Person instance."""
        person = Person(
            name="John Doe",
            index=1,
            voice_embedding=[0.1, 0.2, 0.3],
            cluster_id=1,
            network_id=uuid.UUID(sample_network_id),
        )

        db_session.add(person)
        db_session.commit()
        db_session.refresh(person)

        assert person.id is not None
        assert person.name == "John Doe"  # type: ignore
        assert person.index == 1  # type: ignore
        assert person.voice_embedding == [0.1, 0.2, 0.3]  # type: ignore
        assert person.cluster_id == 1  # type: ignore
        assert person.network_id == uuid.UUID(sample_network_id)  # type: ignore
        assert person.created_at is not None
        assert person.updated_at is not None

    def test_person_unique_index(self, db_session, sample_network_id):
        """Test that person index must be unique."""
        person1 = Person(
            name="John Doe", index=1, network_id=uuid.UUID(sample_network_id)
        )
        person2 = Person(
            name="Jane Doe",
            index=1,
            network_id=uuid.UUID(sample_network_id),  # Same index
        )

        db_session.add(person1)
        db_session.commit()

        db_session.add(person2)
        with pytest.raises(IntegrityError):
            db_session.commit()

    def test_person_relationships(self, db_session, sample_network_id):
        """Test Person relationships."""
        network = MiraNetwork(name="Test Network")
        db_session.add(network)
        db_session.commit()
        db_session.refresh(network)

        person = Person(name="John Doe", index=1, network_id=network.id)

        db_session.add(person)
        db_session.commit()
        db_session.refresh(person)

        # Test relationships
        assert person.network == network
        assert person.interactions == []
        assert person.conversations == []
        assert person.actions == []

    def test_person_optional_fields(self, db_session, sample_network_id):
        """Test creating person with minimal required fields."""
        person = Person(index=1, network_id=uuid.UUID(sample_network_id))

        db_session.add(person)
        db_session.commit()
        db_session.refresh(person)

        assert person.id is not None
        assert person.name is None
        assert person.voice_embedding is None
        assert person.cluster_id is None


class TestConversation:
    """Test cases for Conversation model."""

    def test_create_conversation(self, db_session, sample_network_id):
        """Test creating a Conversation instance."""
        conversation = Conversation(
            user_ids=["user1", "user2"],
            topic_summary="Test conversation",
            context_summary="Test context",
            network_id=uuid.UUID(sample_network_id),
        )

        db_session.add(conversation)
        db_session.commit()
        db_session.refresh(conversation)

        assert conversation.id is not None
        assert conversation.user_ids == ["user1", "user2"]  # type: ignore
        assert conversation.topic_summary == "Test conversation"  # type: ignore
        assert conversation.context_summary == "Test context"  # type: ignore
        assert conversation.network_id == uuid.UUID(sample_network_id)  # type: ignore

    def test_conversation_relationships(self, db_session, sample_network_id):
        """Test Conversation relationships."""
        network = MiraNetwork(name="Test Network")
        db_session.add(network)
        db_session.commit()
        db_session.refresh(network)

        conversation = Conversation(network_id=network.id)

        db_session.add(conversation)
        db_session.commit()
        db_session.refresh(conversation)

        # Test relationships
        assert conversation.network == network
        assert conversation.interactions == []
        assert conversation.persons == []
        assert conversation.actions == []

    def test_conversation_default_values(self, db_session, sample_network_id):
        """Test conversation with default values."""
        conversation = Conversation(network_id=uuid.UUID(sample_network_id))

        db_session.add(conversation)
        db_session.commit()
        db_session.refresh(conversation)

        assert conversation.user_ids == []  # type: ignore
        assert conversation.topic_summary is None
        assert conversation.context_summary is None


class TestInteraction:
    """Test cases for Interaction model."""

    def test_create_interaction(self, db_session, sample_network_id):
        """Test creating an Interaction instance."""
        interaction = Interaction(
            text="Hello, world!",
            voice_embedding=[0.1, 0.2, 0.3],
            text_embedding=[0.4, 0.5, 0.6],
            entities={"PERSON": ["John"]},
            topics=["greeting"],
            sentiment=0.8,
            network_id=uuid.UUID(sample_network_id),
        )

        db_session.add(interaction)
        db_session.commit()
        db_session.refresh(interaction)

        assert interaction.id is not None
        assert interaction.text == "Hello, world!"  # type: ignore
        assert interaction.voice_embedding == [0.1, 0.2, 0.3]  # type: ignore
        assert interaction.text_embedding == [0.4, 0.5, 0.6]  # type: ignore
        assert interaction.entities == {"PERSON": ["John"]}  # type: ignore
        assert interaction.topics == ["greeting"]  # type: ignore
        assert interaction.sentiment == 0.8  # type: ignore
        assert interaction.timestamp is not None

    def test_interaction_relationships(
        self, db_session, sample_network_id, sample_person_id, sample_conversation_id
    ):
        """Test Interaction relationships."""
        network = MiraNetwork(name="Test Network")
        db_session.add(network)
        db_session.commit()
        db_session.refresh(network)

        person = Person(name="John Doe", index=1, network_id=network.id)
        conversation = Conversation(network_id=network.id)

        db_session.add(person)
        db_session.add(conversation)
        db_session.commit()
        db_session.refresh(person)
        db_session.refresh(conversation)

        interaction = Interaction(
            text="Hello!",
            network_id=network.id,
            speaker_id=person.id,
            conversation_id=conversation.id,
        )

        db_session.add(interaction)
        db_session.commit()
        db_session.refresh(interaction)

        # Test relationships
        assert interaction.network == network
        assert interaction.person == person
        assert interaction.conversation == conversation
        assert interaction.actions == []

    def test_interaction_optional_fields(self, db_session, sample_network_id):
        """Test creating interaction with minimal required fields."""
        interaction = Interaction(
            text="Hello!", network_id=uuid.UUID(sample_network_id)
        )

        db_session.add(interaction)
        db_session.commit()
        db_session.refresh(interaction)

        assert interaction.id is not None
        assert interaction.text == "Hello!"  # type: ignore
        assert interaction.conversation_id is None
        assert interaction.speaker_id is None
        assert interaction.voice_embedding is None
        assert interaction.text_embedding is None
        assert interaction.entities is None
        assert interaction.topics is None
        assert interaction.sentiment is None


class TestAction:
    """Test cases for Action model."""

    def test_create_action(self, db_session, sample_network_id, sample_person_id):
        """Test creating an Action instance."""
        action = Action(
            user_id=uuid.uuid4(),
            person_id=uuid.UUID(sample_person_id),
            action_type="send_message",
            details="Send a greeting message",
            status="pending",
            network_id=uuid.UUID(sample_network_id),
        )

        db_session.add(action)
        db_session.commit()
        db_session.refresh(action)

        assert action.id is not None
        assert action.user_id is not None
        assert action.person_id == uuid.UUID(sample_person_id)  # type: ignore
        assert action.action_type == "send_message"  # type: ignore
        assert action.details == "Send a greeting message"  # type: ignore
        assert action.status == "pending"  # type: ignore
        assert action.network_id == uuid.UUID(sample_network_id)  # type: ignore
        assert action.created_at is not None
        assert action.updated_at is not None

    def test_action_relationships(
        self,
        db_session,
        sample_network_id,
        sample_person_id,
        sample_conversation_id,
        sample_interaction_id,
    ):
        """Test Action relationships."""
        network = MiraNetwork(name="Test Network")
        db_session.add(network)
        db_session.commit()
        db_session.refresh(network)

        person = Person(name="John Doe", index=1, network_id=network.id)
        conversation = Conversation(network_id=network.id)
        interaction = Interaction(text="Hello!", network_id=network.id)

        db_session.add(person)
        db_session.add(conversation)
        db_session.add(interaction)
        db_session.commit()
        db_session.refresh(person)
        db_session.refresh(conversation)
        db_session.refresh(interaction)

        action = Action(
            user_id=uuid.uuid4(),
            person_id=person.id,
            action_type="send_message",
            conversation_id=conversation.id,
            interaction_id=interaction.id,
            network_id=network.id,
        )

        db_session.add(action)
        db_session.commit()
        db_session.refresh(action)

        # Test relationships
        assert action.network == network
        assert action.person == person
        assert action.conversation == conversation
        assert action.interaction == interaction

    def test_action_optional_fields(self, db_session, sample_network_id):
        """Test creating action with minimal required fields."""
        action = Action(
            user_id=uuid.uuid4(),
            action_type="test_action",
            network_id=uuid.UUID(sample_network_id),
        )

        db_session.add(action)
        db_session.commit()
        db_session.refresh(action)

        assert action.id is not None
        assert action.person_id is None
        assert action.details is None
        assert action.interaction_id is None
        assert action.conversation_id is None
        assert action.status == "pending"  # type: ignore
        assert action.scheduled_time is None
        assert action.completed_time is None


class TestPersonConversationAssociation:
    """Test cases for person-conversation association table."""

    def test_association_table_structure(self):
        """Test that the association table has the correct structure."""
        assert person_conversation_association.name == "person_conversation"
        assert len(person_conversation_association.columns) == 2

        # Check column names
        column_names = [col.name for col in person_conversation_association.columns]
        assert "person_id" in column_names
        assert "conversation_id" in column_names

    def test_person_conversation_many_to_many(self, db_session, sample_network_id):
        """Test many-to-many relationship between Person and Conversation."""
        network = MiraNetwork(name="Test Network")
        db_session.add(network)
        db_session.commit()
        db_session.refresh(network)

        person1 = Person(name="John", index=1, network_id=network.id)
        person2 = Person(name="Jane", index=2, network_id=network.id)
        conversation = Conversation(network_id=network.id)

        db_session.add(person1)
        db_session.add(person2)
        db_session.add(conversation)
        db_session.commit()
        db_session.refresh(person1)
        db_session.refresh(person2)
        db_session.refresh(conversation)

        # Add persons to conversation
        conversation.persons = [person1, person2]
        db_session.commit()
        db_session.refresh(conversation)

        # Test the relationship
        assert len(conversation.persons) == 2
        assert person1 in conversation.persons
        assert person2 in conversation.persons

        # Test reverse relationship
        assert conversation in person1.conversations
        assert conversation in person2.conversations
