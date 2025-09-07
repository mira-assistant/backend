# API Architecture for Multi-Network System

## Overview

The API architecture provides RESTful endpoints and WebSocket communication for the multi-network Mira system. All endpoints are scoped to Network IDs to ensure data isolation and security.

## API Design Principles

### Network Isolation
- **Network ID Required**: All endpoints require valid Network ID
- **Data Scoping**: All queries automatically filtered by Network ID
- **Authentication**: Network-based authentication and authorization
- **Rate Limiting**: Per-network rate limiting and quotas

### Multi-Client Support
- **Client Registration**: Dynamic client registration and management
- **Real-time Sync**: WebSocket connections for real-time updates
- **Quality Metrics**: Audio quality scoring and stream selection
- **Capability Management**: Client capability negotiation

### Scalability
- **Stateless Design**: Stateless API design for horizontal scaling
- **Caching**: Intelligent caching of network data
- **Load Balancing**: Support for load balancing across instances
- **Database Optimization**: Optimized queries for multi-network scenarios

## REST API Endpoints

### Network Management

#### Create Network
```http
POST /api/v1/networks
Content-Type: application/json

{
  "name": "John's Mira Network",
  "description": "Personal AI assistant network",
  "settings": {
    "audio_quality_threshold": 0.7,
    "max_clients": 10,
    "language": "en"
  }
}
```

**Response:**
```json
{
  "network_id": "net_abc123def456789012345678901234567890",
  "name": "John's Mira Network",
  "status": "active",
  "created_at": "2024-01-15T10:30:00Z",
  "websocket_url": "ws://api.mira.com/ws/net_abc123def456789012345678901234567890",
  "client_registration_token": "client_token_xyz789"
}
```

#### Get Network Details
```http
GET /api/v1/networks/{network_id}
Authorization: Bearer {network_token}
```

#### Update Network Settings
```http
PUT /api/v1/networks/{network_id}
Authorization: Bearer {network_token}
Content-Type: application/json

{
  "name": "Updated Network Name",
  "settings": {
    "audio_quality_threshold": 0.8
  }
}
```

#### Delete Network
```http
DELETE /api/v1/networks/{network_id}
Authorization: Bearer {network_token}
```

### Client Management

#### Register Client
```http
POST /api/v1/networks/{network_id}/clients
Authorization: Bearer {network_token}
Content-Type: application/json

{
  "client_id": "client_789xyz1234567890",
  "device_type": "mobile",
  "device_info": {
    "os": "iOS 17.2",
    "model": "iPhone 15 Pro",
    "version": "1.0.0"
  },
  "capabilities": ["audio", "notifications", "location"],
  "location": {
    "latitude": 37.7749,
    "longitude": -122.4194,
    "accuracy": 10.0
  }
}
```

**Response:**
```json
{
  "client_id": "client_789xyz1234567890",
  "status": "registered",
  "websocket_url": "ws://api.mira.com/ws/net_abc123def456789012345678901234567890",
  "audio_quality_threshold": 0.7,
  "capabilities": ["audio", "notifications", "location"]
}
```

#### Get Network Clients
```http
GET /api/v1/networks/{network_id}/clients
Authorization: Bearer {network_token}
```

#### Update Client Status
```http
PUT /api/v1/clients/{client_id}
Authorization: Bearer {client_token}
Content-Type: application/json

{
  "connection_status": "connected",
  "audio_quality_score": 8.5,
  "location": {
    "latitude": 37.7750,
    "longitude": -122.4195,
    "accuracy": 5.0
  }
}
```

#### Deregister Client
```http
DELETE /api/v1/clients/{client_id}
Authorization: Bearer {client_token}
```

### Audio Processing

#### Submit Audio Buffer
```http
POST /api/v1/networks/{network_id}/audio
Authorization: Bearer {client_token}
Content-Type: multipart/form-data

{
  "audio_data": <binary_audio_data>,
  "client_id": "client_789xyz1234567890",
  "quality_metrics": {
    "snr": 25.5,
    "speech_clarity": 85.2,
    "volume_level": 0.7
  },
  "metadata": {
    "duration": 3.2,
    "sample_rate": 16000,
    "channels": 1
  }
}
```

**Response:**
```json
{
  "processing_id": "proc_abc123def456",
  "status": "processing",
  "estimated_completion": "2024-01-15T10:30:05Z",
  "audio_quality_score": 8.5,
  "selected_for_processing": true
}
```

#### Get Audio Processing Status
```http
GET /api/v1/networks/{network_id}/audio/status
Authorization: Bearer {network_token}
```

#### Get Best Audio Stream
```http
GET /api/v1/networks/{network_id}/audio/best-stream
Authorization: Bearer {network_token}
```

### Interaction Management

#### Get Interactions
```http
GET /api/v1/networks/{network_id}/interactions
Authorization: Bearer {network_token}
Query Parameters:
  - limit: int (default: 50)
  - offset: int (default: 0)
  - client_id: string (optional)
  - speaker_id: string (optional)
  - start_date: datetime (optional)
  - end_date: datetime (optional)
```

#### Get Interaction Details
```http
GET /api/v1/networks/{network_id}/interactions/{interaction_id}
Authorization: Bearer {network_token}
```

#### Process Interaction
```http
POST /api/v1/networks/{network_id}/interactions/{interaction_id}/process
Authorization: Bearer {network_token}
Content-Type: application/json

{
  "context": "Additional context for processing",
  "force_reprocess": false
}
```

### Data Access

#### Get Calendar Events
```http
GET /api/v1/networks/{network_id}/calendar
Authorization: Bearer {network_token}
Query Parameters:
  - start_date: datetime (optional)
  - end_date: datetime (optional)
  - limit: int (default: 50)
```

#### Get Reminders
```http
GET /api/v1/networks/{network_id}/reminders
Authorization: Bearer {network_token}
Query Parameters:
  - status: string (pending, completed, cancelled)
  - due_before: datetime (optional)
  - limit: int (default: 50)
```

#### Get Conversations
```http
GET /api/v1/networks/{network_id}/conversations
Authorization: Bearer {network_token}
Query Parameters:
  - limit: int (default: 20)
  - include_interactions: boolean (default: true)
```

## WebSocket Communication

### Connection Management

#### WebSocket URL Format
```
ws://api.mira.com/ws/{network_id}?token={client_token}
```

#### Connection Handshake
```json
{
  "type": "handshake",
  "client_id": "client_789xyz1234567890",
  "capabilities": ["audio", "notifications"],
  "version": "1.0.0"
}
```

**Server Response:**
```json
{
  "type": "handshake_ack",
  "status": "connected",
  "server_time": "2024-01-15T10:30:00Z",
  "features": ["real_time_audio", "notifications", "context_sync"]
}
```

### WebSocket Events

#### Client → Server Events

##### Audio Data
```json
{
  "type": "audio_data",
  "client_id": "client_789xyz1234567890",
  "audio_buffer": "<base64_encoded_audio>",
  "quality_metrics": {
    "snr": 25.5,
    "speech_clarity": 85.2,
    "volume_level": 0.7
  },
  "timestamp": "2024-01-15T10:30:00Z"
}
```

##### Client Status Update
```json
{
  "type": "client_status",
  "client_id": "client_789xyz1234567890",
  "status": "connected",
  "capabilities": ["audio", "notifications"],
  "location": {
    "latitude": 37.7749,
    "longitude": -122.4194,
    "accuracy": 10.0
  }
}
```

##### Heartbeat
```json
{
  "type": "heartbeat",
  "client_id": "client_789xyz1234567890",
  "timestamp": "2024-01-15T10:30:00Z"
}
```

#### Server → Client Events

##### Interaction Created
```json
{
  "type": "interaction_created",
  "interaction_id": "int_abc123def456",
  "text": "Set a reminder for tomorrow at 3 PM",
  "speaker_id": "person_123",
  "client_id": "client_789xyz1234567890",
  "timestamp": "2024-01-15T10:30:00Z",
  "processing_status": "completed"
}
```

##### Audio Stream Selected
```json
{
  "type": "audio_stream_selected",
  "selected_client_id": "client_789xyz1234567890",
  "quality_score": 8.5,
  "reason": "highest_quality",
  "timestamp": "2024-01-15T10:30:00Z"
}
```

##### Client Joined
```json
{
  "type": "client_joined",
  "client_id": "client_789xyz1234567890",
  "device_type": "mobile",
  "capabilities": ["audio", "notifications"],
  "timestamp": "2024-01-15T10:30:00Z"
}
```

##### Client Left
```json
{
  "type": "client_left",
  "client_id": "client_789xyz1234567890",
  "reason": "disconnected",
  "timestamp": "2024-01-15T10:30:00Z"
}
```

##### Action Executed
```json
{
  "type": "action_executed",
  "action_id": "act_abc123def456",
  "action_type": "create_reminder",
  "result": {
    "reminder_id": "rem_xyz789",
    "title": "Meeting with team",
    "due_time": "2024-01-16T15:00:00Z"
  },
  "timestamp": "2024-01-15T10:30:00Z"
}
```

##### Context Updated
```json
{
  "type": "context_updated",
  "conversation_id": "conv_abc123def456",
  "context_summary": "Discussion about project timeline and deadlines",
  "participants": ["person_123", "person_456"],
  "timestamp": "2024-01-15T10:30:00Z"
}
```

##### Error Notification
```json
{
  "type": "error",
  "error_code": "AUDIO_PROCESSING_FAILED",
  "message": "Failed to process audio buffer",
  "client_id": "client_789xyz1234567890",
  "timestamp": "2024-01-15T10:30:00Z"
}
```

## Authentication and Authorization

### Network Authentication
```python
class NetworkAuthenticator:
    def authenticate_network(self, network_id: str, token: str) -> bool:
        """Authenticate network access"""
        # Validate network exists and is active
        # Validate token matches network
        # Check rate limits and quotas
        pass

    def generate_network_token(self, network_id: str) -> str:
        """Generate authentication token for network"""
        # Create JWT token with network_id claim
        # Set appropriate expiration
        # Include permissions and quotas
        pass
```

### Client Authentication
```python
class ClientAuthenticator:
    def authenticate_client(self, client_id: str, network_id: str, token: str) -> bool:
        """Authenticate client access to network"""
        # Validate client is registered to network
        # Validate token matches client
        # Check client permissions
        pass

    def generate_client_token(self, client_id: str, network_id: str) -> str:
        """Generate authentication token for client"""
        # Create JWT token with client_id and network_id claims
        # Set appropriate expiration
        # Include client capabilities
        pass
```

## Rate Limiting and Quotas

### Network-Level Rate Limiting
```python
class NetworkRateLimiter:
    def __init__(self):
        self.rate_limits = {
            'interactions_per_minute': 100,
            'audio_uploads_per_minute': 50,
            'websocket_connections': 10,
            'api_requests_per_hour': 1000
        }

    def check_rate_limit(self, network_id: str, operation: str) -> bool:
        """Check if network is within rate limits"""
        # Check current usage against limits
        # Update usage counters
        # Return True if within limits
        pass
```

### Client-Level Rate Limiting
```python
class ClientRateLimiter:
    def __init__(self):
        self.rate_limits = {
            'audio_uploads_per_minute': 20,
            'status_updates_per_minute': 10,
            'websocket_messages_per_minute': 100
        }

    def check_client_rate_limit(self, client_id: str, operation: str) -> bool:
        """Check if client is within rate limits"""
        # Check current usage against limits
        # Update usage counters
        # Return True if within limits
        pass
```

## Error Handling

### Error Response Format
```json
{
  "error": {
    "code": "INVALID_NETWORK_ID",
    "message": "The provided network ID is invalid or does not exist",
    "details": {
      "network_id": "invalid_id",
      "timestamp": "2024-01-15T10:30:00Z"
    },
    "request_id": "req_abc123def456"
  }
}
```

### Common Error Codes
- `INVALID_NETWORK_ID`: Network ID not found or invalid
- `INVALID_CLIENT_ID`: Client ID not found or invalid
- `UNAUTHORIZED`: Invalid or missing authentication token
- `RATE_LIMIT_EXCEEDED`: Rate limit exceeded for network or client
- `AUDIO_PROCESSING_FAILED`: Audio processing error
- `MODEL_INFERENCE_FAILED`: AI model inference error
- `WEBSOCKET_CONNECTION_FAILED`: WebSocket connection error

## API Versioning

### Version Header
```http
API-Version: v1
```

### Version Strategy
- **Major Versions**: Breaking changes to API structure
- **Minor Versions**: New features, backward compatible
- **Patch Versions**: Bug fixes, backward compatible

### Deprecation Policy
- **6-month notice** for breaking changes
- **Grace period** for deprecated endpoints
- **Migration guides** for version upgrades

## Monitoring and Analytics

### API Metrics
```python
class APIMetrics:
    def __init__(self):
        self.metrics = {
            'requests_per_network': {},
            'response_times': [],
            'error_rates': {},
            'websocket_connections': 0,
            'audio_processing_volume': 0
        }

    def record_request(self, network_id: str, endpoint: str, duration: float):
        """Record API request metric"""
        pass

    def record_websocket_event(self, network_id: str, event_type: str):
        """Record WebSocket event metric"""
        pass
```

### Health Checks
```http
GET /api/v1/health
```

**Response:**
```json
{
  "status": "healthy",
  "timestamp": "2024-01-15T10:30:00Z",
  "services": {
    "database": "healthy",
    "audio_processing": "healthy",
    "ai_models": "healthy",
    "websocket": "healthy"
  },
  "metrics": {
    "active_networks": 150,
    "active_clients": 450,
    "requests_per_minute": 1200
  }
}
```

This API architecture provides a comprehensive, scalable foundation for the multi-network Mira system while maintaining security, performance, and ease of use.



