# Phone API Endpoints

This document describes the phone-specific API endpoints added for mobile app integration.

## Service Control

### Enable Service
**Endpoint:** `PATCH /phone/service/enable`

Enable the Mira service.

**Response:**
```json
{
  "message": "Mira service enabled successfully",
  "enabled": true
}
```

### Disable Service  
**Endpoint:** `PATCH /phone/service/disable`

Disable the Mira service.

**Response:**
```json
{
  "message": "Mira service disabled successfully", 
  "enabled": false
}
```

### Get Service Status
**Endpoint:** `GET /phone/service/status`

Get current service status and basic information.

**Response:**
```json
{
  "enabled": true,
  "version": "4.1.1",
  "mode": "advanced", 
  "listening_clients": 2,
  "current_best_stream": "client1"
}
```

## Distance Tracking

### Update Phone Distance
**Endpoint:** `POST /phone/distance/update`

Update phone distance to all active clients (affects stream scoring).

**Request Body:**
```json
{
  "distance": 2.5
}
```

**Response:**
```json
{
  "message": "Phone distance updated for 2 clients",
  "distance": 2.5,
  "updated_clients": ["client1", "client2"],
  "current_best_stream": "client1"
}
```

### Get Nearest Client
**Endpoint:** `GET /phone/distance/nearest_client`

Get the client with the shortest phone distance.

**Response:**
```json
{
  "nearest_client": "client1",
  "distance": 1.5,
  "total_clients": 2
}
```

## Database Functions

### Get Recent Interactions
**Endpoint:** `GET /phone/database/interactions/recent?limit=10`

Get recent interactions in simplified format for phone interface.

**Response:**
```json
{
  "interactions": [
    {
      "id": "123e4567-e89b-12d3-a456-426614174000",
      "text": "Hello world",
      "timestamp": "2024-01-01T12:00:00",
      "speaker_id": "speaker-123"
    }
  ],
  "count": 1,
  "limit": 10
}
```

### Clear All Interactions
**Endpoint:** `DELETE /phone/database/interactions/clear?confirm=DELETE_ALL`

Clear all interactions from database (requires confirmation).

**Response:**
```json
{
  "message": "Successfully cleared 25 interactions",
  "deleted_count": 25
}
```

### Get Database Statistics
**Endpoint:** `GET /phone/database/stats`

Get comprehensive database and service statistics.

**Response:**
```json
{
  "database_stats": {
    "total_interactions": 150,
    "total_speakers": 5,
    "total_conversations": 12,
    "last_activity": "2024-01-01T12:00:00"
  },
  "service_status": {
    "enabled": true,
    "listening_clients": 2,
    "best_stream": "client1"
  }
}
```

## Simplified Frontend Integration

The main interaction endpoint has been enhanced to automatically handle stream quality filtering:

### Register Interaction (Enhanced)
**Endpoint:** `POST /interactions/register?client_id=your_client_id`

Send audio data for processing. The backend automatically:
1. Updates stream quality metrics for the client
2. Checks if this client has the best audio stream
3. Only processes the interaction if it's from the best stream
4. Returns rejection message if better streams are available

**If not the best stream:**
```json
{
  "message": "Interaction was not registered due to better audio streams",
  "best_stream_client": "client1", 
  "current_client_score": 65.2,
  "best_stream_score": 78.9
}
```

**If processed normally:**
```json
{
  "id": "interaction-id",
  "text": "transcribed text",
  "timestamp": "2024-01-01T12:00:00",
  "speaker_id": "speaker-id",
  "stream_quality": {
    "client_id": "your_client_id",
    "is_best_stream": true
  }
}
```

This simplified approach means frontends only need to:
1. Use VAD to detect speech
2. Send audio to `/interactions/register` with their client_id  
3. Handle the response (either processed interaction or rejection)

All the complex stream scoring and quality evaluation happens automatically in the backend.