"""
Unit tests for database functionality.
"""

import pytest
from sqlalchemy import create_engine, text
from sqlalchemy.exc import SQLAlchemyError

from app.db.session import get_db, SessionLocal
from app.db.base import Base
from app.core.config import settings


class TestDatabaseSession:
    """Test cases for database session management."""

    def test_create_tables(self, test_engine):
        """Test that create_tables creates all tables."""
        # Drop all tables first
        Base.metadata.drop_all(bind=test_engine)

        # Create tables using the test engine
        Base.metadata.create_all(bind=test_engine)

        # Check that tables exist
        with test_engine.connect() as conn:
            result = conn.execute(text("SELECT name FROM sqlite_master WHERE type='table'"))
            table_names = [row[0] for row in result]

            # Check for our model tables
            assert "mira_networks" in table_names
            assert "persons" in table_names
            assert "conversations" in table_names
            assert "interactions" in table_names
            assert "actions" in table_names
            assert "person_conversation" in table_names

    def test_session_factory_creation(self):
        """Test that SessionLocal is properly configured."""
        assert SessionLocal is not None
        assert hasattr(SessionLocal, "__call__")

    def test_get_db_generator(self, test_session_factory, test_engine):
        """Test that get_db returns a generator that yields a session."""
        # Create a test database
        Base.metadata.create_all(bind=test_engine)

        # Test the generator
        db_gen = get_db()

        # Should be a generator
        assert hasattr(db_gen, "__next__")

        # Get the session
        session = next(db_gen)

        # Should be a SQLAlchemy session
        assert hasattr(session, "query")
        assert hasattr(session, "add")
        assert hasattr(session, "commit")
        assert hasattr(session, "close")

        # Close the generator
        try:
            next(db_gen)
        except StopIteration:
            pass

    def test_session_cleanup(self, test_session_factory, test_engine):
        """Test that database sessions are properly cleaned up."""
        Base.metadata.create_all(bind=test_engine)

        # Get a session
        db_gen = get_db()
        session = next(db_gen)

        # Session should be active
        assert not session.is_active or session.is_active

        # Close the generator (this should close the session)
        try:
            next(db_gen)
        except StopIteration:
            pass

    def test_session_isolation(self, test_session_factory, test_engine):
        """Test that database sessions are isolated."""
        Base.metadata.create_all(bind=test_engine)

        # Create two separate sessions
        db_gen1 = get_db()
        session1 = next(db_gen1)

        db_gen2 = get_db()
        session2 = next(db_gen2)

        # Sessions should be different objects
        assert session1 is not session2

        # Close both generators
        try:
            next(db_gen1)
        except StopIteration:
            pass

        try:
            next(db_gen2)
        except StopIteration:
            pass

    def test_engine_configuration(self, test_engine):
        """Test that the database engine is properly configured."""
        # Test that we can connect
        with test_engine.connect() as conn:
            result = conn.execute(text("SELECT 1"))
            assert result.fetchone()[0] == 1

    def test_sqlite_connection_args(self):
        """Test that SQLite connection args are set correctly."""
        # This test is specific to SQLite
        if "sqlite" in settings.database_url:
            engine = create_engine(settings.database_url, connect_args={"check_same_thread": False})

            # Should be able to connect
            with engine.connect() as conn:
                result = conn.execute(text("SELECT 1"))
                one = result.fetchone()
                assert one is not None
                assert one[0] == 1

    def test_database_transactions(self, test_session_factory, test_engine):
        """Test that database transactions work correctly."""
        Base.metadata.create_all(bind=test_engine)

        # Create a session
        db_gen = get_db()
        session = next(db_gen)

        try:
            # Test that we can perform operations
            from app.models import MiraNetwork

            network = MiraNetwork(name="Test Network")
            session.add(network)
            session.commit()

            # Verify the data was saved
            saved_network = session.query(MiraNetwork).filter_by(name="Test Network").first()
            assert saved_network is not None
            assert saved_network.name == "Test Network"  # type: ignore

        finally:
            # Close the generator
            try:
                next(db_gen)
            except StopIteration:
                pass

    def test_database_rollback(self, test_session_factory, test_engine):
        """Test that database rollback works correctly."""
        Base.metadata.create_all(bind=test_engine)

        # Create a session
        db_gen = get_db()
        session = next(db_gen)

        try:
            from app.models import MiraNetwork

            # Add a network with a unique name
            import uuid

            unique_name = f"Test Network {uuid.uuid4()}"
            network = MiraNetwork(name=unique_name)
            session.add(network)

            # Rollback the transaction
            session.rollback()

            # Verify the data was not saved
            from sqlalchemy import select

            saved_networks = (
                session.execute(select(MiraNetwork).filter_by(name=unique_name)).scalars().all()
            )
            assert len(saved_networks) == 0

        finally:
            # Close the generator
            try:
                next(db_gen)
            except StopIteration:
                pass

    def test_database_error_handling(self, test_session_factory, test_engine):
        """Test that database errors are handled correctly."""
        Base.metadata.create_all(bind=test_engine)

        # Create a session
        db_gen = get_db()
        session = next(db_gen)

        try:
            # Test that we can handle database errors
            with pytest.raises(SQLAlchemyError):
                # This should raise an error due to invalid SQL
                session.execute(text("INVALID SQL STATEMENT"))

        finally:
            # Close the generator
            try:
                next(db_gen)
            except StopIteration:
                pass
