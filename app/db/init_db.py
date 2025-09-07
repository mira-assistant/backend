"""
Database initialization utilities.
"""

from session import create_tables, SessionLocal


def init_db() -> None:
    """Initialize database with tables and default data."""
    # Create tables
    create_tables()

    # Create default data if needed
    db = SessionLocal()
    try:
        # Add any default data here
        pass
    finally:
        db.close()


def reset_db() -> None:
    """Reset database (drop and recreate all tables)."""
    from app.db.base import Base
    from app.db.session import engine

    # Drop all tables
    Base.metadata.drop_all(bind=engine)

    # Recreate tables
    create_tables()

    # Initialize with default data
    init_db()
