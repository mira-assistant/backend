
# Migration Guide: Legacy to New Structure

## Key Changes

### 1. Directory Structure
- **Old**: Flat structure with files in root
- **New**: Organized `app/` package with logical modules

### 2. Database Layer
- **Old**: `db.py` with simple session management
- **New**: `app/db/` with proper session management and Alembic migrations

### 3. Models and Schemas
- **Old**: `models.py` with mixed SQLAlchemy models
- **New**: Separated `app/models/` (SQLAlchemy) and `app/schemas/` (Pydantic)

### 4. API Routes
- **Old**: `routers/` with direct imports
- **New**: `app/api/v1/` with versioned API and proper dependencies

### 5. Business Logic
- **Old**: `processors/` with mixed concerns
- **New**: `app/services/` with clear separation of concerns

## Migration Steps

### 1. Update Imports
Replace old imports with new ones:

```python
# Old
from models import Interaction
from db import get_db_session
from processors.command_processor import CommandProcessor

# New
from app.models.interaction import Interaction
from app.db.session import get_db_session
from app.services.command_service import CommandProcessor
```

### 2. Update Configuration
- Copy `env.example` to `.env`
- Update environment variables as needed
- Use `app.core.config.settings` for configuration

### 3. Database Migration
```bash
# Initialize Alembic (if not done)
alembic init alembic

# Create initial migration
alembic revision --autogenerate -m "Initial migration"

# Apply migrations
alembic upgrade head
```

### 4. Update API Endpoints
- Old endpoints are now under `/api/v1/`
- Authentication is now required for most endpoints
- Use proper HTTP status codes and error handling

### 5. Testing
- Run the new test suite: `pytest`
- Update any custom tests to use new structure

## Backward Compatibility

The new structure maintains backward compatibility where possible:
- Database models have the same structure
- Core functionality remains the same
- API responses maintain the same format

## Support

If you encounter issues during migration:
1. Check the backup files in `legacy_backup/`
2. Review the new API documentation at `/docs`
3. Check the test files for usage examples
