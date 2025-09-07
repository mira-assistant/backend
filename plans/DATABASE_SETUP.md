# Database Setup Guide

This guide explains how to initialize and manage your Mira Backend database using Alembic.

## Prerequisites

1. PostgreSQL database (local or AWS RDS)
2. Python dependencies installed (`pip install -r requirements.txt`)

## Database Initialization

### 1. Configure Database Connection

Update your database URL in one of these ways:

**Option A: Environment Variable**
```bash
export DATABASE_URL="postgresql://username:password@localhost:5432/mira_db"
```

**Option B: .env File**
```bash
# Create .env file in project root
echo "DATABASE_URL=postgresql://username:password@localhost:5432/mira_db" > .env
```

**Option C: Update alembic.ini**
```ini
# In alembic.ini, line 59
sqlalchemy.url = postgresql://username:password@localhost:5432/mira_db
```

### 2. Create Database from Scratch

```bash
# Run all migrations to create the database schema
alembic upgrade head
```

This will:
- Create all tables (mira_networks, persons, conversations, interactions, actions)
- Set up foreign key relationships
- Create indexes and constraints

### 3. Verify Database Setup

```bash
# Check current migration status
alembic current

# View migration history
alembic history

# Show specific migration details
alembic show head
```

## Database Management Commands

### Check Status
```bash
# Current migration version
alembic current

# Migration history
alembic history

# Show specific migration
alembic show <revision_id>
```

### Create New Migrations
```bash
# Auto-generate migration from model changes
alembic revision --autogenerate -m "description of changes"

# Create empty migration
alembic revision -m "description of changes"
```

### Apply Migrations
```bash
# Apply all pending migrations
alembic upgrade head

# Apply specific migration
alembic upgrade <revision_id>

# Apply one migration at a time
alembic upgrade +1
```

### Rollback Migrations
```bash
# Rollback one migration
alembic downgrade -1

# Rollback to specific migration
alembic downgrade <revision_id>

# Rollback all migrations
alembic downgrade base
```

## AWS RDS Setup

### 1. Create RDS PostgreSQL Instance

```bash
# Using AWS CLI
aws rds create-db-instance \
    --db-instance-identifier mira-db \
    --db-instance-class db.t3.micro \
    --engine postgres \
    --master-username mira_user \
    --master-user-password your_secure_password \
    --allocated-storage 20 \
    --vpc-security-group-ids sg-your-security-group \
    --db-subnet-group-name your-subnet-group
```

### 2. Update Connection String

```bash
# For AWS RDS
export DATABASE_URL="postgresql://mira_user:your_secure_password@your-rds-endpoint.region.rds.amazonaws.com:5432/mira_db"
```

### 3. Initialize Database

```bash
# Run migrations on RDS
alembic upgrade head
```

## Troubleshooting

### Common Issues

1. **Connection Refused**
   - Check database is running
   - Verify connection string
   - Check firewall/security groups

2. **Migration Fails**
   - Check database permissions
   - Verify model changes are correct
   - Review migration file for errors

3. **Schema Out of Sync**
   - Check current migration: `alembic current`
   - Compare with expected: `alembic history`
   - Apply missing migrations: `alembic upgrade head`

### Reset Database (Development Only)

```bash
# WARNING: This will delete all data!
# Drop all tables
alembic downgrade base

# Recreate schema
alembic upgrade head
```

## Best Practices

1. **Always use Alembic** for schema changes
2. **Never modify migrations** after they've been applied to production
3. **Test migrations** on a copy of production data
4. **Backup database** before major migrations
5. **Review auto-generated migrations** before applying

## CI/CD Integration

The CI pipeline automatically:
1. Installs dependencies
2. Sets up test database with `alembic upgrade head`
3. Runs tests against the migrated database

No manual database setup is needed for CI/CD.
