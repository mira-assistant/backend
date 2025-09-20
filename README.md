# Mira Backend

A FastAPI-based backend service for the Mira AI assistant, featuring real-time audio processing, multi-stream conversation management, and cloud-native deployment on AWS Lambda with container images.

## 🎯 Project Overview

Mira Backend is a sophisticated AI assistant backend that provides:

- **Real-time Audio Processing**: Advanced audio stream management with noise reduction and quality optimization
- **Multi-Stream Conversation Management**: Handle multiple concurrent audio streams with intelligent conversation routing
- **Person Recognition**: Voice-based person identification and management
- **Scalable Architecture**: Cloud-native design optimized for AWS Lambda with container deployment
- **RESTful API**: Comprehensive API endpoints for frontend integration
- **Database Management**: PostgreSQL integration with Alembic migrations
- **Production Ready**: Built-in monitoring, logging, and security features

### Key Features

- 🎙️ **Audio Stream Processing**: Real-time audio processing with ML-based quality scoring
- 👥 **Person Management**: Voice embedding-based person identification and profile management
- 💬 **Conversation Management**: Multi-participant conversation tracking and management
- 🔄 **Interaction Logging**: Comprehensive interaction recording and retrieval
- 🚀 **Serverless Deployment**: Optimized for AWS Lambda with cold start optimization
- 📊 **Monitoring**: Built-in CloudWatch integration and performance metrics
- 🔒 **Security**: Production-grade security configurations and best practices

### Technology Stack

- **Framework**: FastAPI 0.104+
- **Language**: Python 3.12+
- **Database**: PostgreSQL with SQLAlchemy ORM
- **Migrations**: Alembic
- **AI/ML**: Google Gemini API integration
- **Audio Processing**: Advanced audio processing libraries
- **Deployment**: AWS Lambda + API Gateway + ECR
- **Containerization**: Docker multi-stage builds
- **Monitoring**: AWS CloudWatch

## 🏗️ Architecture

### System Architecture

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   Frontend      │    │   API Gateway   │    │  Lambda Function│
│   (React/Web)   │◄──►│   (AWS)         │◄──►│  (FastAPI)      │
└─────────────────┘    └─────────────────┘    └─────────────────┘
                                                        │
                       ┌─────────────────┐             │
                       │   CloudWatch    │◄────────────┤
                       │   (Monitoring)  │             │
                       └─────────────────┘             │
                                                        │
┌─────────────────┐    ┌─────────────────┐             │
│   ECR Registry  │◄──►│   RDS PostgreSQL│◄────────────┘
│  (Container)    │    │   (Database)    │
└─────────────────┘    └─────────────────┘
```

### Component Overview

- **API Gateway**: HTTP/HTTPS endpoint management and request routing
- **Lambda Function**: Serverless compute running the FastAPI application
- **ECR (Elastic Container Registry)**: Container image storage and versioning
- **RDS PostgreSQL**: Managed database for persistent data storage
- **CloudWatch**: Comprehensive logging and monitoring

## 🚀 Getting Started

### Prerequisites

Before you begin, ensure you have the following installed:

- **Python 3.12+**: Required for development and local testing
- **Docker & Docker Compose**: For containerized development and deployment
- **AWS CLI 2.x**: Configured with appropriate permissions
- **PostgreSQL**: For local development (or use Docker)
- **Git**: For version control

### Quick Start

```bash
# Clone the repository
git clone https://github.com/mira-assistant/backend.git
cd backend

# Copy and edit environment configuration
cp .env.example .env
# Edit .env with your DATABASE_URL and GEMINI_API_KEY

# Start development server
make dev
```

The API will be available at http://localhost:8000

**API Documentation:**
- Swagger UI: http://localhost:8000/docs
- ReDoc: http://localhost:8000/redoc

## 📚 API Reference

### API Versioning

The Mira Backend uses **versioned APIs** to ensure backward compatibility and smooth frontend integration:

- **`/api/v1/`** - Stable production API (recommended for new integrations)
- **`/api/v2/`** - Latest features and enhancements

### Base URL
- **Local Development**: `http://localhost:8000`
- **Production**: `https://your-api-gateway-url.amazonaws.com`

### Authentication
All endpoints use **network-scoped access** with `{network_id}` path parameter for multi-tenancy.

### Complete API Endpoints

#### Service & Network Management
```http
# Register a client to the network
POST /api/v1/{network_id}/service/client/register/{client_id}

# Deregister client from network  
DELETE /api/v1/{network_id}/service/client/deregister/{client_id}

# Get network service information
GET /api/v1/{network_id}/service/info
```

#### Conversation Management
```http
# Get specific conversation
GET /api/v1/{network_id}/conversations/{conversation_id}

# List all conversations (with pagination)
GET /api/v1/{network_id}/conversations?limit=10&offset=0

# Create new conversation
POST /api/v1/{network_id}/conversations
Content-Type: application/json
{
  "title": "string",
  "participants": ["person_id1", "person_id2"]
}

# Update conversation
PUT /api/v1/{network_id}/conversations/{conversation_id}

# Delete conversation
DELETE /api/v1/{network_id}/conversations/{conversation_id}
```

#### Person & Voice Management
```http
# Get person profile
GET /api/v1/{network_id}/persons/{person_id}

# List all persons in network
GET /api/v1/{network_id}/persons?limit=20&offset=0

# Create new person
POST /api/v1/{network_id}/persons
Content-Type: application/json
{
  "name": "string",
  "metadata": {}
}

# Upload voice sample for person
POST /api/v1/{network_id}/persons/{person_id}/voice
Content-Type: multipart/form-data
# Form data: audio_file, description (optional)

# Update person profile
PUT /api/v1/{network_id}/persons/{person_id}

# Delete person
DELETE /api/v1/{network_id}/persons/{person_id}
```

#### Audio Stream Management
```http
# Get currently best audio stream
GET /api/v1/{network_id}/streams/best

# Submit audio stream data
POST /api/v1/{network_id}/streams/{stream_id}/audio
Content-Type: multipart/form-data
# Form data: audio_data, timestamp, metadata (optional)

# List stream quality scores
GET /api/v1/{network_id}/streams/scores

# Get stream by ID
GET /api/v1/{network_id}/streams/{stream_id}
```

#### Interaction & History Management
```http
# Get interaction history (filtered by person)
GET /api/v1/{network_id}/interactions?person_id={person_id}&limit=50&offset=0

# Get interactions for conversation
GET /api/v1/{network_id}/interactions?conversation_id={conversation_id}&limit=50

# Create new interaction record
POST /api/v1/{network_id}/interactions
Content-Type: application/json
{
  "person_id": "uuid",
  "conversation_id": "uuid", 
  "content": "string",
  "interaction_type": "audio|text|system",
  "metadata": {}
}

# Get specific interaction
GET /api/v1/{network_id}/interactions/{interaction_id}

# Update interaction
PUT /api/v1/{network_id}/interactions/{interaction_id}

# Delete interaction
DELETE /api/v1/{network_id}/interactions/{interaction_id}
```

### API v2 Differences

API v2 includes the same endpoints as v1 with these enhancements:
- Improved response schemas with additional metadata
- Enhanced filtering and sorting options
- Better error handling and validation
- New experimental features

Simply replace `/api/v1/` with `/api/v2/` in any endpoint to use the latest version.

### Response Formats

#### Standard Response Schema
```json
{
  "id": "uuid",
  "created_at": "2024-01-01T12:00:00Z",
  "updated_at": "2024-01-01T12:00:00Z",
  "network_id": "uuid"
}
```

#### Conversation Response
```json
{
  "id": "uuid",
  "network_id": "uuid", 
  "title": "Meeting Discussion",
  "created_at": "2024-01-01T12:00:00Z",
  "updated_at": "2024-01-01T12:00:00Z",
  "participants": ["person_id1", "person_id2"],
  "interaction_count": 45
}
```

#### Person Response
```json
{
  "id": "uuid",
  "network_id": "uuid",
  "name": "John Doe", 
  "voice_embedding": "base64_encoded_data",
  "created_at": "2024-01-01T12:00:00Z",
  "conversation_count": 12,
  "metadata": {}
}
```

#### Audio Stream Response
```json
{
  "stream_id": "client_abc_stream_1",
  "quality_score": 0.92,
  "client_id": "client_abc", 
  "timestamp": "2024-01-01T12:00:00Z",
  "is_active": true
}
```

### Error Handling

#### HTTP Status Codes
- `200` Success
- `201` Created
- `400` Bad Request - Invalid input data
- `404` Not Found - Resource doesn't exist
- `422` Validation Error - Schema validation failed
- `500` Internal Server Error

#### Error Response Format
```json
{
  "detail": "Person not found in network",
  "error_code": "PERSON_NOT_FOUND",
  "timestamp": "2024-01-01T12:00:00Z"
}
```

### Interactive Documentation
- **Swagger UI**: http://localhost:8000/docs (recommended for testing)
- **ReDoc**: http://localhost:8000/redoc (better for reading)
- **OpenAPI Spec**: http://localhost:8000/openapi.json

## 🔧 Configuration

### Environment Variables

Create a `.env` file with the following required variables:

```bash
# Database Configuration
DATABASE_URL=postgresql://username:password@localhost:5432/mira_dev

# API Keys  
GEMINI_API_KEY=your_gemini_api_key_here

# Optional Development Settings
DEBUG=true
LOG_LEVEL=INFO
```

### AWS Deployment

For production deployment on AWS Lambda, the application supports:
- Container-based Lambda deployment
- RDS PostgreSQL database
- API Gateway integration
- CloudWatch logging

## 📝 License

This project is licensed under the MIT License.

---

**Built with ❤️ by the Mira Team**

