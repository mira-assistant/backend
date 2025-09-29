# Authentication System Documentation

## Overview

The Mira Backend now includes a comprehensive authentication system that supports:

- Username/password login with JWT tokens
- Google OAuth2 login
- GitHub OAuth2 login  
- Token refresh capabilities
- Backward compatibility with existing network_id-based endpoints

## Authentication Endpoints

### User Registration
```http
POST /api/v1/auth/register
Content-Type: application/json

{
  "username": "optional_username",
  "email": "user@example.com", 
  "password": "secure_password"
}
```

**Response:**
```json
{
  "access_token": "eyJ0eXAiOiJKV1QiLCJ...",
  "refresh_token": "eyJ0eXAiOiJKV1QiLCJ...",
  "token_type": "bearer",
  "user": {
    "id": "uuid",
    "username": "username",
    "email": "user@example.com",
    "is_active": true
  }
}
```

### Username/Password Login
```http
POST /api/v1/auth/login
Content-Type: application/json

{
  "username": "user@example.com",  // Can use email or username
  "password": "secure_password"
}
```

### Google OAuth2 Login
```http
GET /api/v1/auth/google/login
```
Redirects to Google OAuth2. After authorization, user is redirected to:
```http
GET /api/v1/auth/google/callback
```

### GitHub OAuth2 Login  
```http
GET /api/v1/auth/github/login
```
Redirects to GitHub OAuth2. After authorization, user is redirected to:
```http
GET /api/v1/auth/github/callback
```

### Token Refresh
```http
POST /api/v1/auth/refresh
Content-Type: application/json

{
  "refresh_token": "eyJ0eXAiOiJKV1QiLCJ..."
}
```

## Using Authentication

### Making Authenticated Requests

Include the JWT token in the Authorization header:

```http
GET /api/v1/some-endpoint
Authorization: Bearer eyJ0eXAiOiJKV1QiLCJhbGciOiJIUzI1NiJ9...
```

### Backward Compatibility

All existing endpoints continue to work without authentication. If no Authorization header is provided, the system falls back to the original network_id-based behavior.

```http
# This still works exactly as before
GET /api/v1/{network_id}/persons/{person_id}
```

```http  
# This now also works with authentication
GET /api/v1/{network_id}/persons/{person_id}
Authorization: Bearer eyJ0eXAiOiJKV1QiLCJhbGciOiJIUzI1NiJ9...
```

## Configuration

Add these environment variables for OAuth2 support:

```bash
# Required for JWT tokens
SECRET_KEY=your-secret-key-here-change-in-production

# Google OAuth2 (optional)
GOOGLE_CLIENT_ID=your-google-client-id
GOOGLE_CLIENT_SECRET=your-google-client-secret

# GitHub OAuth2 (optional)  
GITHUB_CLIENT_ID=your-github-client-id
GITHUB_CLIENT_SECRET=your-github-client-secret
```

## Token Lifetimes

- **Access tokens**: 30 minutes
- **Refresh tokens**: 7 days

## Database Migration

Run the database migration to add the users table:

```bash
alembic upgrade head
```

## Security Features

- Passwords are hashed using bcrypt
- JWT tokens are signed with HS256
- OAuth2 flows use secure state parameters
- Refresh tokens allow secure token renewal
- User accounts can be disabled
- All endpoints support optional authentication

## Example Integration

Here's how to add optional authentication to an existing endpoint:

```python
from fastapi import Depends
from typing import Optional
from core.dependencies import get_current_user_optional
import models

@router.get("/{network_id}/example")
async def example_endpoint(
    network_id: str,
    current_user: Optional[models.User] = Depends(get_current_user_optional),
    # ... other dependencies
):
    if current_user:
        # User is authenticated - can access user-specific data
        user_id = current_user.id
    else:
        # Fall back to network_id behavior (existing logic)
        network_uuid = uuid.UUID(network_id)
        # ... existing logic
```

This approach ensures minimal changes to existing code while adding powerful authentication capabilities.