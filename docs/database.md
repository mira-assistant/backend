# 🗄️ Database Guide

This guide covers database setup, configuration, and migrations for Mira Backend.

## 📋 Overview

Mira Backend uses:
- **SQLite** - Local development
- **PostgreSQL** - Production (AWS RDS)
- **Alembic** - Database migrations
- **SQLAlchemy** - ORM

## 🚀 Quick Start

### Local Development

```bash
# 1. Activate virtual environment
source venv/bin/activate

# 2. Initialize database (first time only)
python -c "from app.db.init_db import init_db; init_db()"

# 3. Run migrations
alembic upgrade head

# 4. Start application
make dev
```

### Production (AWS RDS)

```bash
# 1. Deploy to AWS (creates RDS)
./scripts/deploy.sh

# 2. Run migrations on RDS
./scripts/migrate-db.sh
```

## 🏗️ Database Schema

### Core Tables

#### `persons`
- **id** (UUID) - Primary key
- **index** (Integer) - Speaker index
- **voice_embedding** (JSON) - Speaker voice profile
- **cluster_id** (Integer) - Speaker cluster assignment

#### `interactions`
- **id** (UUID) - Primary key
- **person_id** (UUID) - Foreign key to persons
- **text** (String) - Transcribed text
- **audio_length** (Float) - Audio duration in seconds
- **timestamp** (DateTime) - When interaction occurred
- **voice_embedding** (JSON) - Voice profile for this interaction

#### `conversations`
- **id** (UUID) - Primary key
- **network_id** (String) - Network identifier
- **title** (String) - Conversation title
- **created_at** (DateTime) - Creation timestamp

#### `networks`
- **id** (String) - Primary key
- **name** (String) - Network name
- **created_at** (DateTime) - Creation timestamp

#### `actions`
- **id** (UUID) - Primary key
- **conversation_id** (UUID) - Foreign key to conversations
- **action_type** (String) - Type of action
- **parameters** (JSON) - Action parameters
- **result** (JSON) - Action result
- **timestamp** (DateTime) - When action occurred

## 🔧 Configuration

### Environment Variables

```bash
# Local development
DATABASE_URL=sqlite:///./mira.db

# Production (AWS RDS)
DATABASE_URL=postgresql://mira_user:password@rds-endpoint:5432/mira_db
```

### Database Settings

Located in `app/core/config.py`:

```python
class Settings(BaseSettings):
    database_url: str = Field(default="sqlite:///./mira.db")

    class Config:
        env_file = ".env"
```

## 📝 Migrations

### Creating Migrations

```bash
# Create new migration
alembic revision --autogenerate -m "Description of changes"

# Create empty migration
alembic revision -m "Description of changes"
```

### Running Migrations

```bash
# Apply all pending migrations
alembic upgrade head

# Apply specific migration
alembic upgrade <revision_id>

# Rollback to previous migration
alembic downgrade -1

# Rollback to specific migration
alembic downgrade <revision_id>
```

### Migration History

```bash
# Show migration history
alembic history

# Show current revision
alembic current

# Show pending migrations
alembic show head
```

## 🛠️ Database Operations

### Using SQLAlchemy

```python
from app.db import get_db_session
from app.models import Person, Interaction

# Get database session
session = get_db_session()

# Create new person
person = Person(
    index=1,
    voice_embedding=[0.1, 0.2, 0.3]  # Example embedding
)
session.add(person)
session.commit()

# Query persons
persons = session.query(Person).all()

# Query with filters
speaker = session.query(Person).filter(Person.index == 1).first()

# Close session
session.close()
```

### Using Dependency Injection

```python
from fastapi import Depends
from app.db import get_db_session

@app.get("/persons")
def get_persons(session: Session = Depends(get_db_session)):
    return session.query(Person).all()
```

## 🔍 Querying Data

### Common Queries

```python
# Get all speakers
speakers = session.query(Person).all()

# Get interactions for a speaker
interactions = session.query(Interaction).filter(
    Interaction.person_id == speaker_id
).all()

# Get recent interactions
recent = session.query(Interaction).order_by(
    Interaction.timestamp.desc()
).limit(10).all()

# Get conversations for a network
conversations = session.query(Conversation).filter(
    Conversation.network_id == network_id
).all()
```

### Advanced Queries

```python
# Join queries
from sqlalchemy.orm import joinedload

# Get person with their interactions
person = session.query(Person).options(
    joinedload(Person.interactions)
).filter(Person.id == person_id).first()

# Aggregate queries
from sqlalchemy import func

# Count interactions per person
counts = session.query(
    Person.index,
    func.count(Interaction.id)
).join(Interaction).group_by(Person.index).all()
```

## 🧪 Testing

### Test Database

```python
# Use in-memory SQLite for tests
DATABASE_URL = "sqlite:///:memory:"

# Or use test database
DATABASE_URL = "sqlite:///./test_mira.db"
```

### Database Fixtures

```python
import pytest
from app.db import get_db_session
from app.models import Person

@pytest.fixture
def db_session():
    session = get_db_session()
    yield session
    session.close()

@pytest.fixture
def sample_person(db_session):
    person = Person(index=1, voice_embedding=[0.1, 0.2])
    db_session.add(person)
    db_session.commit()
    return person
```

## 🚨 Troubleshooting

### Common Issues

#### 1. Migration Conflicts
```bash
# Check current state
alembic current

# Check migration history
alembic history

# Resolve conflicts manually
alembic stamp head
```

#### 2. Connection Issues
```bash
# Test database connection
python -c "from app.db import get_db_session; print('Connected!')"

# Check environment variables
echo $DATABASE_URL
```

#### 3. RDS Connection Issues
```bash
# Check RDS status
aws rds describe-db-instances

# Test connection
psql -h your-rds-endpoint -U mira_user -d mira_db
```

### Debug Commands

```bash
# Check Alembic configuration
alembic show

# Validate migration files
alembic check

# Show SQL for migration
alembic upgrade head --sql
```

## 🔒 Security

### Database Security

1. **Use Environment Variables** - Never hardcode credentials
2. **Connection Pooling** - Configure appropriate pool size
3. **Encryption** - Enable RDS encryption at rest
4. **Network Security** - Use VPC and security groups
5. **Access Control** - Use least-privilege IAM roles

### RDS Security

```bash
# Enable encryption
aws rds modify-db-instance \
  --db-instance-identifier mira-backend-db \
  --storage-encrypted

# Enable automated backups
aws rds modify-db-instance \
  --db-instance-identifier mira-backend-db \
  --backup-retention-period 7
```

## 📊 Performance

### Optimization Tips

1. **Indexes** - Add indexes on frequently queried columns
2. **Connection Pooling** - Configure appropriate pool size
3. **Query Optimization** - Use efficient queries
4. **Caching** - Implement caching for frequently accessed data

### Monitoring

```bash
# Check database size
aws rds describe-db-instances --query 'DBInstances[0].AllocatedStorage'

# Monitor connections
aws cloudwatch get-metric-statistics \
  --namespace AWS/RDS \
  --metric-name DatabaseConnections \
  --dimensions Name=DBInstanceIdentifier,Value=mira-backend-db
```

---

**Need help?** Check the [main README](../README.md) or [deployment guide](deployment.md).
