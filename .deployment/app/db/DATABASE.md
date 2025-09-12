# Database Management Guide

This guide explains how to manage the database for the Mira Backend using Alembic migrations.

## Quick Start

### 1. Initialize Database
```bash
# Activate virtual environment
source .venv/bin/activate

# Initialize database (run migrations)
python scripts/init_db.py

# Start the application
uvicorn app.main:app --reload
```

### 2. Alternative: Manual Migration
```bash
# Run migrations manually
alembic upgrade head

# Start the application
uvicorn app.main:app --reload
```

## Database Schema

The database uses SQLite and includes the following tables:
- `mira_networks` - Network configurations
- `conversations` - Conversation records
- `persons` - Person profiles with voice embeddings
- `interactions` - Individual interaction records
- `person_conversation` - Many-to-many relationship between persons and conversations

## Alembic Commands

### Basic Commands

```bash
# Check current migration status
alembic current

# Show migration history
alembic history

# Show pending migrations
alembic show head
```

### Creating Migrations

```bash
# Auto-generate migration from model changes
alembic revision --autogenerate -m "Description of changes"

# Create empty migration (for manual changes)
alembic revision -m "Description of changes"
```

### Applying Migrations

```bash
# Apply all pending migrations
alembic upgrade head

# Apply specific migration
alembic upgrade <revision_id>

# Apply migrations up to a specific point
alembic upgrade <revision_id>
```

### Rolling Back Migrations

```bash
# Rollback to previous migration
alembic downgrade -1

# Rollback to specific migration
alembic downgrade <revision_id>

# Rollback all migrations
alembic downgrade base
```

## Development Workflow

### 1. Making Model Changes
1. Modify your SQLAlchemy models in `app/models/`
2. Generate migration: `alembic revision --autogenerate -m "Description"`
3. Review the generated migration file
4. Apply migration: `alembic upgrade head`

### 2. Database Reset (Development Only)
```bash
# Remove database file
rm mira.db

# Recreate from scratch
python scripts/init_db.py
```

### 3. Checking Migration Status
```bash
# See current state
alembic current

# See what would be applied
alembic show head
```

## Production Considerations

### 1. Backup Before Migrations
Always backup your database before running migrations in production.

### 2. Test Migrations
Test migrations on a copy of production data before applying to production.

### 3. Migration Order
Migrations are applied in order. Never modify existing migration files that have been applied to production.

### 4. Rollback Plan
Always have a rollback plan for migrations, especially for destructive changes.

## Troubleshooting

### Common Issues

1. **Migration conflicts**: If you have conflicts, resolve them manually in the migration file
2. **Database locked**: Ensure no other processes are using the database
3. **Missing tables**: Run `alembic upgrade head` to apply all migrations

### Reset Everything (Development)
```bash
# Remove database and all migrations
rm mira.db
rm -rf alembic/versions/*

# Create fresh migration
alembic revision --autogenerate -m "Initial migration"
alembic upgrade head
```

## File Structure

```
backend/
├── alembic/
│   ├── versions/          # Migration files
│   ├── env.py            # Alembic environment config
│   └── script.py.mako    # Migration template
├── alembic.ini           # Alembic configuration
├── scripts/
│   └── init_db.py        # Database initialization script
├── app/
│   ├── db/
│   │   ├── base.py       # SQLAlchemy base
│   │   ├── init_db.py    # Database utilities
│   │   └── session.py    # Database session management
│   └── models/           # SQLAlchemy models
└── mira.db               # SQLite database file (created after migration)
```

## Environment Variables

The database URL can be configured via environment variables:
- `DATABASE_URL` - Database connection string (default: `sqlite:///./mira.db`)

Example for different environments:
```bash
# Development
export DATABASE_URL="sqlite:///./mira.db"

# Production (PostgreSQL example)
export DATABASE_URL="postgresql://user:password@localhost/mira_db"
```
