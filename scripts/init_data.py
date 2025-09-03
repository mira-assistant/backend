#!/usr/bin/env python3
"""
Initialize database with sample data.
"""
import sys
import os

# Add the app directory to the Python path
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from app.db.session import SessionLocal
from app.db.init_db import init_db
from app.models.user import User
from app.models.person import Person
from app.core.security import get_password_hash
import uuid


def create_sample_data():
    """Create sample data for development."""
    db = SessionLocal()

    try:
        # Create a sample user
        user = db.query(User).filter(User.username == "admin").first()
        if not user:
            user = User(
                username="admin",
                email="admin@mira.com",
                hashed_password=get_password_hash("admin123"),
                is_active=True,
                is_superuser=True
            )
            db.add(user)
            db.commit()
            print("Created admin user: admin/admin123")

        # Create sample persons
        person1 = db.query(Person).filter(Person.index == 1).first()
        if not person1:
            person1 = Person(
                name="Primary User",
                index=1
            )
            db.add(person1)
            db.commit()
            print("Created primary person")

        person2 = db.query(Person).filter(Person.index == 2).first()
        if not person2:
            person2 = Person(
                name="Secondary User",
                index=2
            )
            db.add(person2)
            db.commit()
            print("Created secondary person")

        print("Sample data initialization complete!")

    except Exception as e:
        print(f"Error creating sample data: {e}")
        db.rollback()
    finally:
        db.close()


if __name__ == "__main__":
    print("Initializing database...")
    init_db()
    print("Creating sample data...")
    create_sample_data()

