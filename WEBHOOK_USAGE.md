# Multi-Tenant Backend API with Webhooks - Usage Guide

This document demonstrates how to use the multi-tenant backend API with JWT authentication and webhook notifications.

## Overview

The enhanced backend now supports:
1. **JWT-based authentication** - All service endpoints require authentication
2. **Network ID extraction from tokens** - No need to manually pass network_id
3. **Webhook registration** - Clients can register webhook URLs to receive interaction updates
4. **Automatic webhook dispatch** - All registered clients receive notifications when new interactions are created

## Authentication

### 1. Register a User

```bash
curl -X POST http://localhost:8000/api/v2/auth/register \
  -H "Content-Type: application/json" \
  -d '{
    "email": "user@example.com",
    "password": "securepassword",
    "username": "myuser"
  }'
```

**Response:**
```json
{
  "access_token": "eyJ0eXAiOiJKV1QiLCJhbGc...",
  "refresh_token": "eyJ0eXAiOiJKV1QiLCJhbGc...",
  "token_type": "bearer",
  "user": {
    "id": "123e4567-e89b-12d3-a456-426614174000",
    "username": "myuser",
    "email": "user@example.com",
    "is_active": true
  }
}
```

### 2. Login

```bash
curl -X POST http://localhost:8000/api/v2/auth/login \
  -H "Content-Type: application/json" \
  -d '{
    "username": "user@example.com",
    "password": "securepassword"
  }'
```

## Client Registration with Webhooks

### Register a Client with Webhook URL

The access token contains the user's ID, which is automatically used as the network_id.

```bash
curl -X POST http://localhost:8000/api/v2/service/client/register/my-client-1 \
  -H "Authorization: Bearer YOUR_ACCESS_TOKEN" \
  -H "Content-Type: application/json" \
  -d '{
    "webhook_url": "https://myapp.example.com/webhook"
  }'
```

**Response:**
```json
{
  "message": "my-client-1 registered successfully",
  "client_id": "my-client-1",
  "webhook_url": "https://myapp.example.com/webhook",
  "registered_at": "2024-01-15T10:30:00Z"
}
```

### Register Without Webhook

You can also register without a webhook URL:

```bash
curl -X POST http://localhost:8000/api/v2/service/client/register/my-client-2 \
  -H "Authorization: Bearer YOUR_ACCESS_TOKEN"
```

## Webhook Payload

When a new interaction is registered, all clients with webhook URLs will receive:

```json
{
  "interaction_id": "456e7890-e89b-12d3-a456-426614174001",
  "network_id": "123e4567-e89b-12d3-a456-426614174000",
  "text": "Hello, this is a test interaction",
  "timestamp": "2024-01-15T10:35:00Z",
  "speaker_id": "789e0123-e89b-12d3-a456-426614174002",
  "conversation_id": "012e3456-e89b-12d3-a456-426614174003"
}
```

### Implementing a Webhook Endpoint

Here's an example Flask webhook receiver:

```python
from flask import Flask, request, jsonify

app = Flask(__name__)

@app.route('/webhook', methods=['POST'])
def receive_webhook():
    data = request.json
    
    print(f"Received interaction: {data['interaction_id']}")
    print(f"Text: {data['text']}")
    print(f"Speaker: {data['speaker_id']}")
    
    # Process the interaction data
    # ... your business logic here ...
    
    return jsonify({"status": "received"}), 200

if __name__ == '__main__':
    app.run(port=5000)
```

## Service Management

### Enable Service

```bash
curl -X PATCH http://localhost:8000/api/v2/service/enable \
  -H "Authorization: Bearer YOUR_ACCESS_TOKEN"
```

### Disable Service

```bash
curl -X PATCH http://localhost:8000/api/v2/service/disable \
  -H "Authorization: Bearer YOUR_ACCESS_TOKEN"
```

### Deregister Client

```bash
curl -X DELETE http://localhost:8000/api/v2/service/client/deregister/my-client-1 \
  -H "Authorization: Bearer YOUR_ACCESS_TOKEN"
```

## Token Security

The JWT tokens include:
- **Signature verification** - Uses HS256 algorithm with secret key
- **Expiration** - Access tokens expire after 30 minutes
- **Type checking** - Validates token type (access vs refresh)

Example token payload:
```json
{
  "sub": "123e4567-e89b-12d3-a456-426614174000",
  "exp": 1705318200,
  "type": "access"
}
```

## Interaction Registration Flow

1. **Client registers** with webhook URL
2. **Client sends audio/interaction** to `/api/v2/{network_id}/interactions/register`
3. **Backend processes** the interaction (transcription, speaker identification, etc.)
4. **Backend saves** interaction to database
5. **Backend dispatches webhooks** to all registered clients
6. **Clients receive** webhook notifications and process updates

## Error Handling

### Authentication Errors

```json
{
  "detail": "Invalid or expired token"
}
```

**HTTP Status:** 401 Unauthorized

### Client Not Found

```json
{
  "detail": "Client my-client-1 not found"
}
```

**HTTP Status:** 404 Not Found

### Invalid Webhook URL

```json
{
  "detail": [
    {
      "type": "url_parsing",
      "loc": ["body", "webhook_url"],
      "msg": "Input should be a valid URL",
      "input": "not-a-valid-url"
    }
  ]
}
```

**HTTP Status:** 422 Unprocessable Entity

## Best Practices

1. **Store tokens securely** - Never expose access tokens in client-side code
2. **Implement webhook retries** - Backend attempts webhook delivery once; implement your own retry logic
3. **Use HTTPS for webhooks** - Always use secure URLs for production webhooks
4. **Validate webhook payloads** - Verify the payload structure on your webhook endpoint
5. **Handle webhook failures gracefully** - Backend continues processing even if webhooks fail
6. **Refresh tokens before expiry** - Use the refresh token endpoint before access token expires

## Architecture Benefits

### Multi-Tenancy
- Each user has their own isolated network (identified by user ID)
- JWT tokens automatically associate requests with the correct network
- No manual network_id passing required

### Scalability
- Webhooks sent asynchronously with concurrent HTTP requests
- Failed webhooks don't block interaction processing
- HTTP client connection pooling for efficiency

### Security
- Industry-standard JWT with signature verification
- Token expiration enforced
- User authentication required for all operations
- Network isolation per user

## Example Integration

Complete example with Python `requests`:

```python
import requests

# Base URL
BASE_URL = "http://localhost:8000/api/v2"

# 1. Register user
response = requests.post(f"{BASE_URL}/auth/register", json={
    "email": "developer@example.com",
    "password": "mypassword",
    "username": "developer"
})
token_data = response.json()
access_token = token_data["access_token"]

# 2. Register client with webhook
headers = {"Authorization": f"Bearer {access_token}"}
response = requests.post(
    f"{BASE_URL}/service/client/register/webclient",
    headers=headers,
    json={"webhook_url": "https://myapp.example.com/webhook"}
)
print(f"Client registered: {response.json()}")

# 3. Enable service
response = requests.patch(
    f"{BASE_URL}/service/enable",
    headers=headers
)
print(f"Service enabled: {response.json()}")

# Now when interactions are created, your webhook will receive notifications!
```

## Migration from Old API

### Before (v1 - No Authentication)
```bash
POST /api/v1/{network_id}/service/client/register/{client_id}
# network_id must be provided in URL
```

### After (v2 - With Authentication)
```bash
POST /api/v2/service/client/register/{client_id}
Authorization: Bearer <token>
# network_id extracted from token automatically
```

The v2 API provides better security and simpler client integration by eliminating the need to manage network IDs separately from user authentication.
