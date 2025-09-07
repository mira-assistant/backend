# Mira Network Architecture Plans

This directory contains comprehensive documentation and planning for the Mira Network-based AI Assistant system. The documentation is organized into focused areas covering the transition from a single-user local system to a multi-network, multi-client architecture.

## Documentation Structure

### Core Architecture
- **[Network Structure](./01-network-structure.md)** - Network-based account system and multi-client architecture
- **[Database Design](./02-database-design.md)** - Database schema for multi-network support
- **[API Architecture](./03-api-architecture.md)** - REST API and WebSocket endpoints

### Processing Pipeline
- **[Inference Pipeline](./04-inference-pipeline.md)** - AI model processing and action extraction
- **[Audio Processing](./05-audio-processing.md)** - Multi-stream audio handling and quality selection
- **[Context Management](./06-context-management.md)** - Conversation context and speaker recognition

### Implementation
- **[Migration Plan](./07-migration-plan.md)** - Step-by-step migration from legacy to network architecture
- **[Security Model](./08-security-model.md)** - Authentication, authorization, and data isolation
- **[Performance Considerations](./09-performance-considerations.md)** - Scalability and optimization strategies

### Development
- **[Development Guidelines](./10-development-guidelines.md)** - Coding standards and best practices
- **[Testing Strategy](./11-testing-strategy.md)** - Testing approach for multi-network system
- **[Deployment Guide](./12-deployment-guide.md)** - Production deployment and monitoring

## Key Concepts

### Network-Based Architecture
Mira uses a **Network** as the primary account entity instead of traditional user accounts. Each network can have multiple connected clients (devices) that work together as a unified system.

### Multi-Client Audio Processing
The system intelligently selects the best audio stream from multiple connected devices based on quality metrics including SNR, speech clarity, and device location.

### Real-time Synchronization
All clients in a network receive real-time updates about interactions, context changes, and system status through WebSocket connections.

### Data Isolation
Each network's data is completely isolated using Network IDs, ensuring privacy and security between different user networks.

## Legacy System Analysis

The current legacy system (in `/legacy_code/`) was designed for single-user local development and includes:

- **Processors**: Audio processing, context management, inference, command processing
- **Models**: Person, Interaction, Conversation, Action entities
- **Routers**: Service, interaction, conversation, persons, streams management
- **Database**: SQLite with basic relationships

The migration plan outlines how to transform this into a scalable, multi-network system while preserving the core AI processing capabilities.

## Getting Started

1. Review the [Network Structure](./01-network-structure.md) to understand the overall architecture
2. Examine the [Database Design](./02-database-design.md) for data modeling
3. Follow the [Migration Plan](./07-migration-plan.md) for implementation steps
4. Refer to [Development Guidelines](./10-development-guidelines.md) for coding standards

## Status

- ✅ Legacy system analysis completed
- ✅ Network architecture designed
- ✅ Database schema planned
- 🔄 Implementation in progress
- ⏳ Testing strategy pending
- ⏳ Deployment planning pending



