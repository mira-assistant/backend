import os
import logging
from contextlib import contextmanager
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.exc import SQLAlchemyError, IntegrityError, DatabaseError
from models import Base

logger = logging.getLogger(__name__)

DB_PATH = os.path.join(os.path.dirname(__file__), "mira.db")
DATABASE_URL = f"sqlite:///{DB_PATH}"

engine = create_engine(DATABASE_URL, echo=False)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

try:
    Base.metadata.create_all(bind=engine)
except SQLAlchemyError as e:
    logger.error(f"Database initialization failed: {e}")
    raise


def get_db_session():
    return SessionLocal()


@contextmanager
def get_db_session_context():
    session = SessionLocal()
    try:
        yield session
        session.commit()
    except Exception:
        session.rollback()
        raise
    finally:
        session.close()


class DatabaseError(Exception):
    pass


class DatabaseIntegrityError(DatabaseError):
    pass


class DatabaseConnectionError(DatabaseError):
    pass


def handle_db_error(operation_name: str):
    def decorator(func):
        def wrapper(*args, **kwargs):
            try:
                return func(*args, **kwargs)
            except IntegrityError as e:
                logger.error(f"Database integrity error in {operation_name}: {e}")
                raise DatabaseIntegrityError(f"Data integrity violation in {operation_name}")
            except DatabaseError as e:
                logger.error(f"Database connection error in {operation_name}: {e}")
                raise DatabaseConnectionError(f"Database connection failed in {operation_name}")
            except SQLAlchemyError as e:
                logger.error(f"Database error in {operation_name}: {e}")
                raise DatabaseError(f"Database operation failed in {operation_name}")
        return wrapper
    return decorator
