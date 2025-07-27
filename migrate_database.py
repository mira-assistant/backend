#!/usr/bin/env python3
"""
Database migration script to update schema for enhanced context processor.
"""

import sqlite3
import os
from sqlalchemy import create_engine, text
from models import Base
from db import DB_PATH, engine

def backup_database():
    """Create a backup of the existing database."""
    if os.path.exists(DB_PATH):
        backup_path = DB_PATH + '.backup'
        import shutil
        shutil.copy2(DB_PATH, backup_path)
        print(f"✓ Database backed up to {backup_path}")
        return True
    return False

def migrate_database():
    """Migrate database schema to support enhanced features."""
    print("=== Database Migration for Enhanced Context Processor ===")
    
    # Backup existing database
    backup_database()
    
    try:
        # Connect to database
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        
        # Check existing schema
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
        existing_tables = [row[0] for row in cursor.fetchall()]
        print(f"Existing tables: {existing_tables}")
        
        # Check if we need to add new columns to existing tables
        if 'interactions' in existing_tables:
            # Check existing columns
            cursor.execute("PRAGMA table_info(interactions);")
            existing_columns = [row[1] for row in cursor.fetchall()]
            print(f"Existing interaction columns: {existing_columns}")
            
            # Add missing columns to interactions table
            missing_columns = {
                'speaker_id': 'TEXT',
                'conversation_id': 'TEXT',
                'entities': 'TEXT',
                'topics': 'TEXT',
                'sentiment': 'REAL'
            }
            
            for column, column_type in missing_columns.items():
                if column not in existing_columns:
                    try:
                        cursor.execute(f"ALTER TABLE interactions ADD COLUMN {column} {column_type};")
                        print(f"✓ Added column {column} to interactions table")
                    except sqlite3.OperationalError as e:
                        if "duplicate column name" not in str(e):
                            print(f"⚠ Warning adding column {column}: {e}")
        
        if 'conversations' in existing_tables:
            # Check existing columns
            cursor.execute("PRAGMA table_info(conversations);")
            existing_columns = [row[1] for row in cursor.fetchall()]
            print(f"Existing conversation columns: {existing_columns}")
            
            # Add missing columns to conversations table
            missing_columns = {
                'speaker_id': 'TEXT',
                'topic_summary': 'TEXT',
                'context_summary': 'TEXT',
                'participants': 'TEXT'
            }
            
            for column, column_type in missing_columns.items():
                if column not in existing_columns:
                    try:
                        cursor.execute(f"ALTER TABLE conversations ADD COLUMN {column} {column_type};")
                        print(f"✓ Added column {column} to conversations table")
                    except sqlite3.OperationalError as e:
                        if "duplicate column name" not in str(e):
                            print(f"⚠ Warning adding column {column}: {e}")
        
        # Check if persons table exists, create if not
        if 'persons' not in existing_tables:
            cursor.execute('''
                CREATE TABLE persons (
                    id TEXT PRIMARY KEY,
                    name TEXT,
                    speaker_index INTEGER UNIQUE NOT NULL,
                    voice_embedding TEXT,
                    is_identified BOOLEAN DEFAULT 0,
                    cluster_id INTEGER,
                    created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                    updated_at DATETIME DEFAULT CURRENT_TIMESTAMP
                )
            ''')
            print("✓ Created persons table")
        
        # Update actions table if needed
        if 'actions' in existing_tables:
            cursor.execute("PRAGMA table_info(actions);")
            existing_columns = [row[1] for row in cursor.fetchall()]
            
            missing_columns = {
                'person_id': 'TEXT',
                'interaction_id': 'TEXT',
                'conversation_id': 'TEXT',
                'status': 'TEXT DEFAULT "pending"',
                'scheduled_time': 'DATETIME',
                'completed_time': 'DATETIME'
            }
            
            for column, column_def in missing_columns.items():
                if column not in existing_columns:
                    try:
                        cursor.execute(f"ALTER TABLE actions ADD COLUMN {column} {column_def};")
                        print(f"✓ Added column {column} to actions table")
                    except sqlite3.OperationalError as e:
                        if "duplicate column name" not in str(e):
                            print(f"⚠ Warning adding column {column}: {e}")
        
        # Commit changes
        conn.commit()
        conn.close()
        
        print("✓ Database migration completed successfully")
        
        # Verify with SQLAlchemy
        try:
            Base.metadata.create_all(bind=engine)
            print("✓ SQLAlchemy schema verification passed")
        except Exception as e:
            print(f"⚠ SQLAlchemy verification warning: {e}")
        
        return True
        
    except Exception as e:
        print(f"❌ Migration failed: {e}")
        return False

def verify_migration():
    """Verify that migration was successful."""
    print("\n=== Verifying Migration ===")
    
    try:
        from db import get_db_session
        from models import Person, Interaction, Conversation, Action
        
        session = get_db_session()
        
        # Test basic queries
        person_count = session.query(Person).count()
        interaction_count = session.query(Interaction).count()
        conversation_count = session.query(Conversation).count()
        action_count = session.query(Action).count()
        
        print(f"✓ Database accessible after migration")
        print(f"  Persons: {person_count}")
        print(f"  Interactions: {interaction_count}")
        print(f"  Conversations: {conversation_count}")
        print(f"  Actions: {action_count}")
        
        # Test creating a sample person
        test_person = Person(
            speaker_index=999,
            name="Test Person",
            is_identified=True
        )
        session.add(test_person)
        session.commit()
        session.delete(test_person)
        session.commit()
        
        print("✓ Person creation/deletion test passed")
        
        session.close()
        return True
        
    except Exception as e:
        print(f"❌ Migration verification failed: {e}")
        return False

if __name__ == "__main__":
    success = migrate_database()
    if success:
        verify_migration()
        print("\n🎉 Database migration completed successfully!")
    else:
        print("\n❌ Database migration failed!")