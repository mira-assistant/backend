# Network Structure Architecture

## Overview

Mira uses a **Network-based account system** where each user creates a "Network" that acts as their account. All devices connected to this network work together to create a unified Mira experience, with data separated by Network IDs and synchronized across all connected clients.

## Core Concepts

### Network (Account)
- **Primary Entity**: A Network represents a user's account
- **Unique Identifier**: Each network has a unique Network ID (e.g., `net_abc123def456`)
- **Data Isolation**: All data is scoped to a specific Network ID
- **Multi-Device**: Supports unlimited connected clients/devices

### Client (Device)
- **Device Registration**: Each device (phone, laptop, tablet) registers as a client
- **Unique Client ID**: Each client has a unique identifier within the network
- **Capabilities**: Clients can have different capabilities (audio, notifications, etc.)
- **Quality Scoring**: Audio quality is scored to select the best stream

### Network Manager
- **Central Orchestrator**: Manages all networks and their clients
- **Audio Selection**: Chooses the best audio stream from available clients
- **Real-time Sync**: Broadcasts updates to all clients in a network
- **Data Access**: Provides network-scoped data access

## Architecture Diagram

```
┌─────────────────────────────────────────────────────────────┐
│                    MIRA NETWORK SYSTEM                     │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  ┌─────────────────┐    ┌─────────────────────────────────┐ │
│  │   USER DEVICE   │    │         BACKEND SERVER         │ │
│  │                 │    │                                 │ │
│  │ ┌─────────────┐ │    │ ┌─────────────────────────────┐ │ │
│  │ │   Client A  │◄┼────┼─┤      Network Manager        │ │ │
│  │ │ (Phone)     │ │    │ │                             │ │ │
│  │ └─────────────┘ │    │ │ • Network ID: abc123        │ │ │
│  │                 │    │ │ • Client Registry           │ │ │
│  │ ┌─────────────┐ │    │ │ • Audio Stream Selector    │ │ │
│  │ │   Client B  │◄┼────┼─┤ • WebSocket Notifications  │ │ │
│  │ │ (Laptop)    │ │    │ └─────────────────────────────┘ │ │
│  │ └─────────────┘ │    │                                 │ │
│  │                 │    │ ┌─────────────────────────────┐ │ │
│  │ ┌─────────────┐ │    │ │      Database Layer        │ │ │
│  │ │   Client C  │◄┼────┼─┤                             │ │ │
│  │ │ (Tablet)    │ │    │ │ • Network Data (abc123)     │ │ │
│  │ └─────────────┘ │    │ │ • Calendar Entries         │ │ │
│  │                 │    │ │ • Reminders                 │ │ │
│  │                 │    │ │ • Interactions              │ │ │
│  │                 │    │ │ • Client States             │ │ │
│  │                 │    │ └─────────────────────────────┘ │ │
│  └─────────────────┘    └─────────────────────────────────┘ │
└─────────────────────────────────────────────────────────────┘
```

## Network Lifecycle

### 1. Network Creation
```
User Downloads App
        ↓
   Create New Network
        ↓
   Generate Network ID
        ↓
   Register First Client
        ↓
   Network Ready for Use
```

### 2. Client Registration
```
Client Connects
        ↓
   Authenticate with Network ID
        ↓
   Register Client Capabilities
        ↓
   Join Audio Stream Pool
        ↓
   Receive WebSocket Connection
        ↓
   Client Active in Network
```

### 3. Audio Processing Flow
```
Multiple Clients Recording
        ↓
   Audio Quality Scoring
        ↓
   Best Stream Selection
        ↓
   Process & Log Interaction
        ↓
   Broadcast to All Clients
        ↓
   Update Network Context
```

## Network Manager Components

### Network Registry
- **Network Storage**: Maintains active networks and their metadata
- **Client Registry**: Tracks all clients within each network
- **Connection Management**: Handles client connections and disconnections

### Audio Stream Selector
- **Quality Metrics**: Evaluates SNR, speech clarity, volume, location
- **Real-time Scoring**: Continuously scores all active audio streams
- **Best Stream Selection**: Chooses optimal stream for processing

### WebSocket Manager
- **Real-time Communication**: Manages WebSocket connections per network
- **Event Broadcasting**: Sends updates to all clients in a network
- **Connection Health**: Monitors and manages connection states

### Data Access Layer
- **Network Scoping**: All queries filtered by Network ID
- **Data Isolation**: Ensures complete separation between networks
- **Caching**: Optimizes data access for multi-client scenarios

## Client Capabilities

### Audio Capabilities
- **Recording**: Capture audio from microphone
- **Quality Metrics**: Provide audio quality scores
- **Stream Management**: Handle audio stream lifecycle

### Notification Capabilities
- **Real-time Updates**: Receive WebSocket notifications
- **UI Updates**: Update interface based on network events
- **Status Sync**: Synchronize state across devices

### Data Capabilities
- **Local Storage**: Cache network data locally
- **Sync Management**: Handle data synchronization
- **Offline Support**: Work with cached data when offline

## Network States

### Active Network
- **Status**: Fully operational with connected clients
- **Audio Processing**: Active audio stream selection
- **Real-time Sync**: WebSocket connections active
- **Data Access**: Full access to network data

### Inactive Network
- **Status**: No active clients connected
- **Audio Processing**: Paused
- **Real-time Sync**: WebSocket connections closed
- **Data Access**: Read-only access to historical data

### Suspended Network
- **Status**: Temporarily disabled (user action or system issue)
- **Audio Processing**: Disabled
- **Real-time Sync**: Connections closed
- **Data Access**: Limited access

## Benefits of Network Architecture

### Seamless Multi-Device Experience
- All devices work together as one unified system
- Data automatically synchronized across all clients
- Best audio quality automatically selected

### Scalable and Flexible
- Easy to add/remove devices from network
- Network can grow with user needs
- Independent of specific device types

### Data Privacy and Security
- Complete isolation between different networks
- User controls all their data
- No shared data between different users

### Real-time Collaboration
- All clients notified of changes instantly
- Shared context across devices
- Unified interaction history

## Network ID Format

### Structure
```
net_[32-character-hex-string]
Example: net_abc123def456789012345678901234567890
```

### Generation
- **Algorithm**: Cryptographically secure random generation
- **Length**: 32 characters (128 bits)
- **Uniqueness**: Globally unique across all networks
- **Collision Resistance**: Extremely low probability of collision

## Client ID Format

### Structure
```
client_[16-character-hex-string]
Example: client_789xyz1234567890
```

### Generation
- **Algorithm**: Cryptographically secure random generation
- **Length**: 16 characters (64 bits)
- **Scope**: Unique within a network
- **Reusability**: Can be reused after client disconnection

## Network Metadata

### Network Information
```json
{
  "network_id": "net_abc123def456",
  "name": "John's Mira Network",
  "created_at": "2024-01-15T10:30:00Z",
  "updated_at": "2024-01-15T14:22:00Z",
  "status": "active",
  "client_count": 3,
  "last_activity": "2024-01-15T14:22:00Z"
}
```

### Client Information
```json
{
  "client_id": "client_789xyz1234567890",
  "network_id": "net_abc123def456",
  "device_type": "mobile",
  "device_info": {
    "os": "iOS 17.2",
    "model": "iPhone 15 Pro",
    "version": "1.0.0"
  },
  "capabilities": ["audio", "notifications"],
  "connection_status": "connected",
  "last_seen": "2024-01-15T14:22:00Z",
  "audio_quality_score": 8.5
}
```

## Implementation Considerations

### State Management
- **Network State**: Centralized state for each network
- **Client State**: Per-client state tracking
- **Sync State**: Synchronization status across clients

### Error Handling
- **Network Errors**: Handle network-level failures
- **Client Errors**: Manage individual client failures
- **Sync Errors**: Handle synchronization conflicts

### Performance Optimization
- **Caching**: Intelligent caching of network data
- **Batch Operations**: Batch database operations
- **Connection Pooling**: Efficient WebSocket management

This network-based architecture provides a robust foundation for a multi-device AI assistant that feels like a single, cohesive system while maintaining clear data boundaries and security.



