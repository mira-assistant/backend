# Mira Backend

A production-ready FastAPI-based backend service for the Mira AI assistant, featuring real-time audio processing, multi-stream conversation management, and cloud-native deployment on AWS Lambda with container images.

## 📋 Table of Contents

- [Project Overview](#-project-overview)
- [Architecture](#-architecture)
- [Getting Started](#-getting-started)
- [API Reference](#-api-reference)
- [AWS Lambda Deployment](#-aws-lambda-deployment)
- [Container Deployment](#-container-deployment)
- [Configuration](#-configuration)
- [Development](#-development)
- [Monitoring & Logging](#-monitoring--logging)
- [Security](#-security)
- [Contributing](#-contributing)
- [License](#-license)

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

### Installation & Setup

#### 1. Clone and Setup Repository

```bash
# Clone the repository
git clone https://github.com/mira-assistant/backend.git
cd backend

# Copy environment configuration
cp .env.example .env

# Edit .env with your configuration
nano .env  # or your preferred editor
```

#### 2. Environment Configuration

Update your `.env` file with the following required variables:

```bash
# Database Configuration
DATABASE_URL=postgresql://username:password@localhost:5432/mira_dev
DB_PASSWORD=your_secure_password

# API Keys
GEMINI_API_KEY=your_gemini_api_key_here

# Optional: Development Settings
DEBUG=true
LOG_LEVEL=DEBUG
```

#### 3. Local Development Setup

**Option A: Native Python Development**

```bash
# Create virtual environment
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Run database migrations
make db-migrate

# Start development server
make dev
```

**Option B: Docker Development (Recommended)**

```bash
# Build and run development container
make dev-build
make dev-run

# The API will be available at http://localhost:8000
```

**Option C: Docker Compose (Full Stack)**

```bash
# Start all services (API + Database)
make docker-compose-up

# View logs
docker-compose logs -f

# Stop services
docker-compose down
```

#### 4. Verify Installation

Once the server is running, verify the installation:

```bash
# Check API health
curl http://localhost:8000/

# View API documentation
open http://localhost:8000/docs  # Swagger UI
open http://localhost:8000/redoc  # ReDoc
```

### Quick Test

Create a test network and verify the API:

```bash
# Create a test network (replace with actual endpoint)
curl -X POST "http://localhost:8000/api/v1/networks" \
  -H "Content-Type: application/json" \
  -d '{"name": "test-network"}'

# List available endpoints
curl http://localhost:8000/api/v1/
```

## 📚 API Reference

The Mira Backend provides a comprehensive RESTful API with two versions (v1 and v2) for maximum compatibility and feature evolution.

### Base URLs

- **Local Development**: `http://localhost:8000`
- **Production**: `https://your-api-gateway-url.amazonaws.com`

### Authentication

Currently, the API uses network-based authentication. All endpoints require a valid `network_id` parameter.

### API Versions

- **v1**: Stable API for production use
- **v2**: Latest features and improvements

### Core Endpoints

#### Network Management

**Get Network Information**
```http
GET /api/v1/{network_id}/service/info
```

**Register Client**
```http
POST /api/v1/{network_id}/service/client/register/{client_id}
```

**Deregister Client**
```http
DELETE /api/v1/{network_id}/service/client/deregister/{client_id}
```

#### Conversation Management

**Get Conversation**
```http
GET /api/v1/{network_id}/conversations/{conversation_id}
```

Response:
```json
{
  "id": "uuid",
  "network_id": "uuid",
  "title": "string",
  "created_at": "datetime",
  "updated_at": "datetime",
  "participants": ["person_id1", "person_id2"]
}
```

**List Conversations**
```http
GET /api/v1/{network_id}/conversations?limit=10&offset=0
```

#### Person Management

**Get Person Profile**
```http
GET /api/v1/{network_id}/persons/{person_id}
```

Response:
```json
{
  "id": "uuid",
  "network_id": "uuid",
  "name": "string",
  "voice_embedding": "base64_encoded_data",
  "created_at": "datetime",
  "conversation_count": "integer"
}
```

**Upload Person Voice Sample**
```http
POST /api/v1/{network_id}/persons/{person_id}/voice
Content-Type: multipart/form-data
```

Parameters:
- `audio_file`: Audio file (WAV, MP3, FLAC)
- `description`: Optional description

#### Audio Stream Management

**Get Best Audio Stream**
```http
GET /api/v1/{network_id}/streams/best
```

Response:
```json
{
  "stream_id": "string",
  "quality_score": "float",
  "client_id": "string",
  "timestamp": "datetime"
}
```

**Submit Audio Stream**
```http
POST /api/v1/{network_id}/streams/{stream_id}/audio
Content-Type: multipart/form-data
```

Parameters:
- `audio_data`: Raw audio data
- `timestamp`: Audio timestamp
- `metadata`: Optional metadata object

#### Interaction Management

**Get Interaction History**
```http
GET /api/v1/{network_id}/interactions?person_id={person_id}&limit=50
```

**Create Interaction Record**
```http
POST /api/v1/{network_id}/interactions
Content-Type: application/json
```

Request Body:
```json
{
  "person_id": "uuid",
  "conversation_id": "uuid",
  "content": "string",
  "interaction_type": "audio|text|system",
  "metadata": {}
}
```

### Error Responses

The API uses standard HTTP status codes and returns detailed error information:

```json
{
  "detail": "Error description",
  "error_code": "SPECIFIC_ERROR_CODE",
  "timestamp": "2024-01-01T00:00:00Z"
}
```

#### Common Status Codes

- `200`: Success
- `201`: Created
- `400`: Bad Request
- `401`: Unauthorized
- `404`: Not Found
- `422`: Validation Error
- `500`: Internal Server Error

### Rate Limiting

The API implements rate limiting to ensure fair usage:

- **Development**: 1000 requests per hour per IP
- **Production**: Configurable based on plan

### Interactive API Documentation

- **Swagger UI**: Available at `/docs`
- **ReDoc**: Available at `/redoc`
- **OpenAPI Spec**: Available at `/openapi.json`

## ☁️ AWS Lambda Deployment

Mira Backend is optimized for serverless deployment on AWS Lambda with container images, providing automatic scaling, cost efficiency, and high availability.

### Deployment Architecture

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   GitHub        │    │   GitHub        │    │   AWS ECR       │
│   Repository    │───►│   Actions       │───►│   Container     │
└─────────────────┘    └─────────────────┘    │   Registry      │
                                              └─────────────────┘
                                                        │
                                                        ▼
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   CloudWatch    │◄──►│   API Gateway   │◄──►│   Lambda        │
│   Monitoring    │    │   HTTP API      │    │   Function      │
└─────────────────┘    └─────────────────┘    └─────────────────┘
                                                        │
                                                        ▼
                                              ┌─────────────────┐
                                              │   RDS           │
                                              │   PostgreSQL    │
                                              └─────────────────┘
```

### Prerequisites for AWS Deployment

1. **AWS Account**: With appropriate permissions
2. **AWS CLI**: Version 2.x configured
3. **Terraform** (Optional): For infrastructure as code
4. **GitHub Repository**: With Actions enabled

### Step-by-Step Deployment Guide

#### 1. AWS Infrastructure Setup

**Option A: Using AWS Console (Manual)**

1. **Create ECR Repository**:
   ```bash
   aws ecr create-repository \
     --repository-name mira-backend \
     --region us-east-1
   ```

2. **Create RDS PostgreSQL Instance**:
   ```bash
   aws rds create-db-instance \
     --db-instance-identifier mira-db \
     --db-instance-class db.t3.micro \
     --engine postgres \
     --master-username mira_admin \
     --master-user-password YourSecurePassword123! \
     --allocated-storage 20 \
     --vpc-security-group-ids sg-xxxxxxxxx
   ```

3. **Create Lambda Execution Role**:
   - Create IAM role with Lambda, ECR, RDS, and CloudWatch permissions
   - Attach policies: `AWSLambdaExecute`, `AmazonRDSDataFullAccess`

**Option B: Using Terraform (Recommended)**

```bash
# Navigate to infrastructure directory
cd scripts/deployment/terraform

# Initialize Terraform
terraform init

# Plan deployment
terraform plan -var-file="production.tfvars"

# Apply infrastructure
terraform apply -var-file="production.tfvars"
```

#### 2. Environment Configuration

**Set up GitHub Secrets** (for GitHub Actions deployment):

Navigate to your GitHub repository → Settings → Secrets and variables → Actions

Add the following secrets:

```
AWS_ACCESS_KEY_ID=AKIA...
AWS_SECRET_ACCESS_KEY=...
AWS_REGION=us-east-1
ECR_REPOSITORY_URI=123456789012.dkr.ecr.us-east-1.amazonaws.com/mira-backend
DATABASE_URL=postgresql://mira_admin:password@mira-db.xxx.us-east-1.rds.amazonaws.com:5432/mira_prod
GEMINI_API_KEY=your_gemini_api_key_here
```

#### 3. Automated Deployment (GitHub Actions)

The repository includes GitHub Actions workflows for automated deployment:

**Deployment Workflow** (`.github/workflows/deploy.yml`):

```yaml
name: Deploy to AWS Lambda

on:
  push:
    branches: [main, production]
  pull_request:
    branches: [main]

jobs:
  deploy:
    runs-on: ubuntu-latest
    steps:
      - name: Checkout code
        uses: actions/checkout@v4
      
      - name: Configure AWS credentials
        uses: aws-actions/configure-aws-credentials@v4
        with:
          aws-access-key-id: ${{ secrets.AWS_ACCESS_KEY_ID }}
          aws-secret-access-key: ${{ secrets.AWS_SECRET_ACCESS_KEY }}
          aws-region: ${{ secrets.AWS_REGION }}
      
      - name: Build and push container
        run: |
          make deploy-container
      
      - name: Update Lambda function
        run: |
          make deploy-lambda
```

**Trigger Deployment**:

```bash
# Deploy to staging
git push origin main

# Deploy to production
git push origin production
```

#### 4. Manual Deployment

For manual deployment or troubleshooting:

**Build and Push Container**:

```bash
# Login to ECR
aws ecr get-login-password --region us-east-1 | \
  docker login --username AWS --password-stdin \
  123456789012.dkr.ecr.us-east-1.amazonaws.com

# Build Lambda container
make lambda-build

# Tag and push
docker tag mira-api:lambda \
  123456789012.dkr.ecr.us-east-1.amazonaws.com/mira-backend:latest

docker push 123456789012.dkr.ecr.us-east-1.amazonaws.com/mira-backend:latest
```

**Create/Update Lambda Function**:

```bash
# Create new function
aws lambda create-function \
  --function-name mira-backend \
  --package-type Image \
  --code ImageUri=123456789012.dkr.ecr.us-east-1.amazonaws.com/mira-backend:latest \
  --role arn:aws:iam::123456789012:role/lambda-execution-role \
  --timeout 300 \
  --memory-size 512

# Update existing function
aws lambda update-function-code \
  --function-name mira-backend \
  --image-uri 123456789012.dkr.ecr.us-east-1.amazonaws.com/mira-backend:latest
```

#### 5. Database Migration

**Run database migrations on AWS**:

```bash
# Using Lambda function directly
aws lambda invoke \
  --function-name mira-backend \
  --payload '{"action": "run_migrations"}' \
  response.json

# Or using the make command
make db-migrate-aws
```

#### 6. API Gateway Setup

**Create HTTP API**:

```bash
aws apigatewayv2 create-api \
  --name mira-backend-api \
  --protocol-type HTTP \
  --target arn:aws:lambda:us-east-1:123456789012:function:mira-backend
```

**Configure Routes**:

```bash
# Create catch-all route
aws apigatewayv2 create-route \
  --api-id abcdef123 \
  --route-key 'ANY /{proxy+}' \
  --target integrations/integration-id
```

### Deployment Verification

After deployment, verify everything is working:

```bash
# Test API Gateway endpoint
curl https://abcdef123.execute-api.us-east-1.amazonaws.com/

# Check Lambda logs
aws logs tail /aws/lambda/mira-backend --follow

# Verify database connection
curl https://abcdef123.execute-api.us-east-1.amazonaws.com/health
```

### Performance Optimization

**Lambda Configuration**:

- **Memory**: 512MB-1GB (adjust based on usage)
- **Timeout**: 300 seconds (5 minutes max)
- **Provisioned Concurrency**: For consistent performance
- **Environment Variables**: Minimal set for cold start optimization

**Container Optimization**:

- Multi-stage builds to minimize image size
- Lambda-specific optimizations in Dockerfile.lambda
- Dependency layer caching

### Monitoring and Alerting

Set up CloudWatch alarms for:

- Lambda function errors
- API Gateway 4xx/5xx errors
- Database connection failures
- Cold start duration

```bash
# Create error rate alarm
aws cloudwatch put-metric-alarm \
  --alarm-name "Lambda-ErrorRate" \
  --alarm-description "Lambda function error rate" \
  --metric-name Errors \
  --namespace AWS/Lambda \
  --statistic Average \
  --period 300 \
  --threshold 5 \
  --comparison-operator GreaterThanThreshold
```

### Rollback Strategy

**Quick Rollback**:

```bash
# Revert to previous container version
aws lambda update-function-code \
  --function-name mira-backend \
  --image-uri 123456789012.dkr.ecr.us-east-1.amazonaws.com/mira-backend:previous

# Or use versioning
aws lambda update-alias \
  --function-name mira-backend \
  --name LIVE \
  --function-version 1
```

## 🐳 Container Deployment

## 📁 Project Structure

```
mira-backend/
├── app/                      # FastAPI application core
│   ├── api/                  # API route definitions
│   │   ├── v1/              # API version 1 (stable)
│   │   │   ├── conversation_router.py  # Conversation management
│   │   │   ├── interaction_router.py   # Interaction logging
│   │   │   ├── persons_router.py       # Person/voice management
│   │   │   ├── service_router.py       # Network/client services
│   │   │   └── streams_router.py       # Audio stream handling
│   │   ├── v2/              # API version 2 (latest features)
│   │   └── __init__.py      # API package initialization
│   ├── core/                # Core application components
│   │   ├── config.py        # Configuration management
│   │   └── mira_logger.py   # Logging configuration
│   ├── db/                  # Database management
│   │   ├── DATABASE.md      # Database documentation
│   │   ├── __init__.py      # Database initialization
│   │   └── session.py       # Database session management
│   ├── models/              # SQLAlchemy database models
│   │   ├── action.py        # Action model
│   │   ├── conversation.py  # Conversation model
│   │   ├── interaction.py   # Interaction model
│   │   ├── network.py       # Network model
│   │   ├── person.py        # Person model
│   │   └── __init__.py      # Models package
│   ├── services/            # Business logic services
│   │   ├── service_factory.py  # Service factory pattern
│   │   └── __init__.py      # Services package
│   ├── tests/               # Test suite
│   │   ├── integration/     # Integration tests
│   │   ├── unit/           # Unit tests
│   │   └── test_main.py    # Main application tests
│   └── main.py             # FastAPI application entry point
├── docker/                  # Docker configurations
│   ├── Dockerfile.dev      # Development container
│   └── Dockerfile.lambda   # Production Lambda container
├── scripts/                # Utility and deployment scripts
│   ├── deployment/         # AWS deployment scripts
│   │   ├── terraform/      # Infrastructure as Code
│   │   └── deploy.sh       # Deployment automation
│   └── init_db.py         # Database initialization
├── alembic/                # Database migration management
│   ├── versions/           # Migration files
│   ├── env.py             # Alembic environment
│   └── script.py.mako     # Migration template
├── .github/                # GitHub Actions workflows
│   └── workflows/          # CI/CD pipelines
├── requirements.txt        # Python dependencies
├── pyproject.toml         # Project configuration and tools
├── alembic.ini            # Alembic configuration
├── Makefile               # Development automation
├── .env.example           # Environment variable template
├── docker-compose.yml     # Local development stack
└── README.md              # This documentation
```

### Key Components Explained

#### **`app/api/`** - API Route Handlers
- **Versioned APIs**: Separate v1 and v2 for backward compatibility
- **Modular routers**: Each domain (conversations, persons, etc.) has its own router
- **Path parameters**: Network-scoped endpoints for multi-tenancy

#### **`app/models/`** - Database Models
- **SQLAlchemy ORM**: Type-safe database interactions
- **UUID primary keys**: Globally unique identifiers
- **Relationship mapping**: Foreign keys and associations

#### **`app/services/`** - Business Logic
- **Service layer**: Separates business logic from API handlers
- **Factory pattern**: Flexible service instantiation
- **Dependency injection**: Clean separation of concerns

#### **`docker/`** - Container Configurations
- **Multi-environment**: Separate containers for dev and production
- **Optimized builds**: Minimal production images with security hardening

#### **`alembic/`** - Database Migrations
- **Version control**: Track database schema changes
- **Automated migrations**: Deploy-time database updates
- **Rollback support**: Safe schema versioning

## 🛠️ Development

### Development Workflow

#### **1. Setting Up Development Environment**

```bash
# Clone repository
git clone https://github.com/mira-assistant/backend.git
cd backend

# Setup environment
cp .env.example .env
# Edit .env with your configuration

# Choose development method:
# Option A: Native Python
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# Option B: Docker (recommended)
make dev-build
```

#### **2. Database Development**

```bash
# Start PostgreSQL (if using Docker)
docker-compose up -d postgres

# Run migrations
make db-migrate

# Create new migration (after model changes)
alembic revision --autogenerate -m "Description of changes"

# Apply migrations
alembic upgrade head

# Reset database (development only)
make db-reset
```

#### **3. Code Development Cycle**

```bash
# Start development server with hot reload
make dev

# In another terminal, run tests
make test

# Run linting and formatting
make lint
make format

# Type checking (if using mypy)
mypy app/
```

#### **4. API Development**

```bash
# View API documentation
open http://localhost:8000/docs  # Swagger UI
open http://localhost:8000/redoc # ReDoc

# Test API endpoints
curl -X GET http://localhost:8000/health
curl -X GET http://localhost:8000/api/v1/{network_id}/conversations
```

### Available Make Commands

#### **Development Commands**
```bash
make dev              # Build and run development server
make dev-build        # Build development Docker image
make dev-run          # Run development container
make install          # Install Python dependencies
make shell            # Open development shell
```

#### **Testing Commands**
```bash
make test             # Run all tests
make test-unit        # Run unit tests only
make test-integration # Run integration tests only
make test-coverage    # Run tests with coverage report
make test-watch       # Run tests in watch mode
```

#### **Code Quality Commands**
```bash
make lint             # Run linting (flake8, black check)
make format           # Format code (isort, black)
make type-check       # Run type checking (mypy)
make security-check   # Run security scanning
make audit            # Full code audit (lint + security + type)
```

#### **Database Commands**
```bash
make db-migrate       # Run database migrations
make db-migrate-aws   # Run migrations on AWS RDS
make db-reset         # Reset database (development only)
make db-seed          # Seed database with test data
make db-backup        # Backup database
make db-restore       # Restore database from backup
```

#### **Docker Commands**
```bash
make docker-build     # Build development image
make docker-build-lambda  # Build Lambda production image
make docker-run       # Run development container
make docker-test      # Test container deployment
make docker-push      # Push image to registry
make docker-clean     # Clean up Docker resources
```

#### **Deployment Commands**
```bash
make deploy           # Full deployment to AWS
make deploy-infra     # Deploy infrastructure only
make deploy-app       # Deploy application only
make deploy-container # Build and push container
make deploy-lambda    # Update Lambda function
```

### Development Best Practices

#### **Code Style**
- **Black**: Code formatting (88 character line length)
- **isort**: Import sorting and organization
- **flake8**: Linting and style checking
- **Type hints**: Use Python type annotations

#### **Testing**
- **pytest**: Test framework with fixtures
- **Coverage**: Minimum 80% test coverage
- **Unit tests**: Fast, isolated component tests
- **Integration tests**: End-to-end API testing

#### **Database**
- **Alembic migrations**: Always create migrations for schema changes
- **Model validation**: Use Pydantic for request/response validation
- **Connection pooling**: Efficient database connection management

#### **API Design**
- **REST principles**: Consistent HTTP methods and status codes
- **Versioning**: Backward-compatible API evolution
- **Documentation**: Auto-generated OpenAPI documentation
- **Error handling**: Structured error responses

#### **Security**
- **Environment variables**: Never commit secrets
- **Input validation**: Validate all user inputs
- **SQL injection**: Use ORM parameterized queries
- **CORS**: Configure appropriate cross-origin policies

### Debugging

#### **Local Debugging**

```bash
# Run with debugger
python -m debugpy --listen 5678 --wait-for-client app/main.py

# Debug with VS Code
# Add breakpoints and use "Python: Remote Attach" configuration
```

#### **Container Debugging**

```bash
# Run container in debug mode
docker run -it --rm -p 8000:8000 -p 5678:5678 \
  -v $(PWD)/app:/app \
  mira-api:dev \
  python -m debugpy --listen 0.0.0.0:5678 --wait-for-client -m uvicorn main:app --host 0.0.0.0
```

#### **Production Debugging**

```bash
# View Lambda logs
aws logs tail /aws/lambda/mira-backend --follow

# Debug specific request
aws logs filter-log-events \
  --log-group-name /aws/lambda/mira-backend \
  --filter-pattern "ERROR"
```

### Performance Optimization

#### **Local Performance**
- **Async/await**: Use async handlers for I/O operations
- **Connection pooling**: Optimize database connections
- **Caching**: Implement appropriate caching strategies

#### **Lambda Performance**
- **Cold start optimization**: Minimize initialization time
- **Memory allocation**: Right-size Lambda memory
- **Provisioned concurrency**: For consistent performance

### Troubleshooting

#### **Common Issues**

1. **Database connection errors**:
   ```bash
   # Check database status
   make db-status
   
   # Reset database
   make db-reset
   ```

2. **Container build failures**:
   ```bash
   # Clean Docker cache
   make docker-clean
   
   # Rebuild from scratch
   make docker-build --no-cache
   ```

3. **Import errors**:
   ```bash
   # Check Python path
   export PYTHONPATH="${PYTHONPATH}:$(pwd)/app"
   
   # Verify virtual environment
   which python
   ```

4. **Port conflicts**:
   ```bash
   # Find process using port 8000
   lsof -i :8000
   
   # Kill process
   kill -9 <PID>
   ```

This project uses **container-based deployment** to handle complex ML dependencies and ensure consistent environments across development and production.

### Container Architecture

The project includes two optimized Dockerfiles:

- **`docker/Dockerfile.dev`**: Development container with debugging tools
- **`docker/Dockerfile.lambda`**: Production container optimized for AWS Lambda

### Development Container

**Build and Run Development Container**:

```bash
# Build development image
make dev-build

# Run with live code reloading
make dev-run

# Alternative: Use Docker Compose for full stack
docker-compose up -d
```

**Development Features**:
- Hot code reloading
- Debug tools included
- Volume mounting for live editing
- Development database included

### Production Container

**Build Lambda-Optimized Container**:

```bash
# Build production image
make lambda-build

# Test locally (simulates Lambda environment)
make lambda-run

# Test container deployment
make docker-test
```

**Production Optimizations**:
- Multi-stage build for minimal image size
- Lambda Runtime Interface Client (RIC)
- Optimized Python dependencies
- Security hardening

### Container Registry Management

**Push to ECR**:

```bash
# Login to Amazon ECR
aws ecr get-login-password --region us-east-1 | \
  docker login --username AWS --password-stdin \
  your-account.dkr.ecr.us-east-1.amazonaws.com

# Tag and push
docker tag mira-api:lambda \
  your-account.dkr.ecr.us-east-1.amazonaws.com/mira-backend:latest

docker push your-account.dkr.ecr.us-east-1.amazonaws.com/mira-backend:latest
```

### Container Health Checks

All containers include health check endpoints:

```bash
# Check container health
curl http://localhost:8000/health

# Detailed health check
curl http://localhost:8000/health/detailed
```

## 🔧 Configuration

### Environment Variables

The application uses environment-based configuration for flexibility and security.

#### Required Variables

```bash
# Database Configuration
DATABASE_URL=postgresql://username:password@host:port/database
DB_PASSWORD=secure_password_here

# AI/ML API Keys
GEMINI_API_KEY=your_gemini_api_key_here
HUGGINGFACE_API_KEY=your_huggingface_key_here  # Optional

# Application Settings
APP_NAME=Mira Backend
APP_VERSION=5.0.0
DEBUG=false  # Set to true for development
LOG_LEVEL=INFO  # DEBUG, INFO, WARNING, ERROR, CRITICAL
```

#### Optional Variables

```bash
# CORS Configuration
CORS_ORIGINS=*  # Comma-separated list for production

# AWS Configuration (auto-detected in Lambda)
AWS_REGION=us-east-1
AWS_ACCESS_KEY_ID=your_access_key
AWS_SECRET_ACCESS_KEY=your_secret_key

# Performance Tuning
MAX_WORKERS=4
TIMEOUT_KEEP_ALIVE=65
```

### Environment File Setup

**Development Environment** (`.env`):

```bash
# Copy example and customize
cp .env.example .env

# Edit configuration
nano .env
```

**Production Environment**:

```bash
# Use environment variables in deployment
export DATABASE_URL="postgresql://prod_user:secure_pass@prod-db:5432/mira"
export GEMINI_API_KEY="production_api_key"
export DEBUG="false"
export LOG_LEVEL="WARNING"
```

### AWS Configuration

#### Lambda Environment Variables

Set in AWS Lambda console or via deployment:

```bash
aws lambda update-function-configuration \
  --function-name mira-backend \
  --environment "Variables={
    DATABASE_URL=postgresql://user:pass@rds-endpoint:5432/mira,
    GEMINI_API_KEY=your_key,
    LOG_LEVEL=INFO
  }"
```

#### RDS Configuration

**Database Setup**:

```sql
-- Create database and user
CREATE DATABASE mira_prod;
CREATE USER mira_user WITH PASSWORD 'secure_password';
GRANT ALL PRIVILEGES ON DATABASE mira_prod TO mira_user;

-- Create required extensions
\c mira_prod
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";
```

**Connection Parameters**:

```bash
# Standard PostgreSQL connection
DATABASE_URL=postgresql://mira_user:password@mira-db.region.rds.amazonaws.com:5432/mira_prod

# With SSL (recommended for production)
DATABASE_URL=postgresql://mira_user:password@mira-db.region.rds.amazonaws.com:5432/mira_prod?sslmode=require
```

#### Security Group Configuration

**RDS Security Group** (Allow Lambda access):

```bash
# Allow Lambda subnet access to RDS
aws ec2 authorize-security-group-ingress \
  --group-id sg-rds-security-group \
  --protocol tcp \
  --port 5432 \
  --source-group sg-lambda-security-group
```

### Configuration Validation

**Validate Configuration**:

```bash
# Check environment variables
python -c "from app.core.config import settings; print(settings.dict())"

# Test database connection
python -c "from app.db import get_db; next(get_db())"

# Test API keys
curl -X GET "http://localhost:8000/health/detailed"
```

### Configuration Management Best Practices

1. **Never commit secrets**: Use `.env` files locally, environment variables in production
2. **Use strong passwords**: Minimum 16 characters with mixed case, numbers, and symbols
3. **Rotate API keys**: Regularly rotate external API keys
4. **Environment separation**: Different configurations for dev/staging/production
5. **Least privilege**: Grant minimal required permissions

## 📊 Monitoring & Logging

Comprehensive monitoring and observability for production deployments.

### CloudWatch Integration

#### **Lambda Function Monitoring**

```bash
# View real-time logs
aws logs tail /aws/lambda/mira-backend --follow

# Query logs for specific patterns
aws logs filter-log-events \
  --log-group-name /aws/lambda/mira-backend \
  --filter-pattern "ERROR" \
  --start-time $(date -d '1 hour ago' +%s)000

# Get function metrics
aws cloudwatch get-metric-statistics \
  --namespace AWS/Lambda \
  --metric-name Duration \
  --dimensions Name=FunctionName,Value=mira-backend \
  --start-time $(date -d '1 hour ago' --iso-8601) \
  --end-time $(date --iso-8601) \
  --period 300 \
  --statistics Average,Maximum
```

#### **API Gateway Monitoring**

```bash
# Monitor API Gateway metrics
aws cloudwatch get-metric-statistics \
  --namespace AWS/ApiGatewayV2 \
  --metric-name Count \
  --dimensions Name=ApiId,Value=your-api-id \
  --start-time $(date -d '1 hour ago' --iso-8601) \
  --end-time $(date --iso-8601) \
  --period 300 \
  --statistics Sum
```

#### **RDS Database Monitoring**

```bash
# Database connection metrics
aws cloudwatch get-metric-statistics \
  --namespace AWS/RDS \
  --metric-name DatabaseConnections \
  --dimensions Name=DBInstanceIdentifier,Value=mira-db \
  --start-time $(date -d '1 hour ago' --iso-8601) \
  --end-time $(date --iso-8601) \
  --period 300 \
  --statistics Average,Maximum
```

### Application Logging

#### **Structured Logging**

The application uses structured JSON logging for better observability:

```python
# Example log output
{
  "timestamp": "2024-01-01T12:00:00Z",
  "level": "INFO",
  "logger": "mira.api.v1.conversation",
  "message": "Conversation retrieved successfully",
  "request_id": "uuid-request-id",
  "user_id": "uuid-user-id",
  "network_id": "uuid-network-id",
  "duration_ms": 45,
  "status_code": 200
}
```

#### **Log Levels and Configuration**

```bash
# Set log level via environment variable
export LOG_LEVEL=DEBUG  # DEBUG, INFO, WARNING, ERROR, CRITICAL

# Configure log format
export LOG_FORMAT=json  # json, text
```

### Key Metrics

#### **Application Metrics**

| Metric | Description | Threshold |
|--------|-------------|-----------|
| Request Duration | API response time | < 500ms (95th percentile) |
| Error Rate | 4xx/5xx error percentage | < 1% |
| Database Connections | Active DB connections | < 80% of max |
| Memory Usage | Lambda memory utilization | < 80% |
| Cold Start Rate | Lambda cold start frequency | < 5% |

#### **Business Metrics**

| Metric | Description | Dashboard |
|--------|-------------|-----------|
| Active Conversations | Number of ongoing conversations | Real-time |
| Audio Stream Quality | Average stream quality score | Hourly |
| Person Recognition Accuracy | Voice ID accuracy rate | Daily |
| API Usage by Endpoint | Request distribution | Real-time |

### Alerting Configuration

#### **CloudWatch Alarms**

```bash
# High error rate alarm
aws cloudwatch put-metric-alarm \
  --alarm-name "MiraBackend-HighErrorRate" \
  --alarm-description "Lambda function error rate > 5%" \
  --metric-name Errors \
  --namespace AWS/Lambda \
  --statistic Sum \
  --period 300 \
  --threshold 5 \
  --comparison-operator GreaterThanThreshold \
  --dimensions Name=FunctionName,Value=mira-backend \
  --evaluation-periods 2 \
  --alarm-actions arn:aws:sns:us-east-1:123456789012:alerts

# High duration alarm
aws cloudwatch put-metric-alarm \
  --alarm-name "MiraBackend-HighDuration" \
  --alarm-description "Lambda duration > 10 seconds" \
  --metric-name Duration \
  --namespace AWS/Lambda \
  --statistic Average \
  --period 300 \
  --threshold 10000 \
  --comparison-operator GreaterThanThreshold \
  --dimensions Name=FunctionName,Value=mira-backend
```

#### **Database Alerts**

```bash
# Database connection alarm
aws cloudwatch put-metric-alarm \
  --alarm-name "MiraDB-HighConnections" \
  --alarm-description "Database connections > 80%" \
  --metric-name DatabaseConnections \
  --namespace AWS/RDS \
  --statistic Average \
  --period 300 \
  --threshold 16 \
  --comparison-operator GreaterThanThreshold \
  --dimensions Name=DBInstanceIdentifier,Value=mira-db
```

### Health Check Endpoints

```bash
# Basic health check
curl http://localhost:8000/health
# Response: {"status": "healthy", "timestamp": "2024-01-01T12:00:00Z"}

# Detailed health check
curl http://localhost:8000/health/detailed
# Response: {
#   "status": "healthy",
#   "checks": {
#     "database": "healthy",
#     "external_apis": "healthy",
#     "memory_usage": "normal"
#   },
#   "version": "5.0.0",
#   "uptime": 3600
# }
```

### Dashboard Configuration

#### **CloudWatch Dashboard**

Create a comprehensive dashboard for monitoring:

```json
{
  "widgets": [
    {
      "type": "metric",
      "properties": {
        "metrics": [
          ["AWS/Lambda", "Duration", "FunctionName", "mira-backend"],
          ["AWS/Lambda", "Errors", "FunctionName", "mira-backend"],
          ["AWS/Lambda", "Invocations", "FunctionName", "mira-backend"]
        ],
        "period": 300,
        "stat": "Average",
        "region": "us-east-1",
        "title": "Lambda Performance"
      }
    }
  ]
}
```

### Performance Monitoring

#### **Application Performance Monitoring (APM)**

For advanced monitoring, consider integrating with:

- **AWS X-Ray**: Distributed tracing
- **Datadog**: Full-stack monitoring
- **New Relic**: Application performance monitoring

#### **Custom Metrics**

```python
# Example: Custom business metrics
import boto3

cloudwatch = boto3.client('cloudwatch')

def publish_custom_metric(metric_name, value, unit='Count'):
    cloudwatch.put_metric_data(
        Namespace='Mira/Application',
        MetricData=[
            {
                'MetricName': metric_name,
                'Value': value,
                'Unit': unit,
                'Dimensions': [
                    {
                        'Name': 'Environment',
                        'Value': 'production'
                    }
                ]
            }
        ]
    )

# Usage
publish_custom_metric('ConversationsCreated', 1)
publish_custom_metric('AudioStreamQuality', 0.95, 'Percent')
```

## 🔒 Security

Production-grade security implementation and best practices.

### Authentication & Authorization

#### **Network-Based Authentication**

Currently implements network-scoped access control:

```python
# All endpoints require valid network_id
@router.get("/{network_id}/conversations/{conversation_id}")
def get_conversation(
    network_id: str = Path(..., description="The ID of the network"),
    conversation_id: str = Path(..., description="The ID of the conversation"),
    db: Session = Depends(get_db),
):
    # Validates network exists and user has access
    # Scopes all operations to the specific network
```

#### **Future Authentication Options**

Planned authentication mechanisms:

- **JWT Tokens**: Stateless authentication
- **API Keys**: Service-to-service authentication
- **OAuth 2.0**: Third-party integration
- **IAM Roles**: AWS-native authentication

### Data Security

#### **Database Security**

```bash
# PostgreSQL security configuration
# Enable SSL/TLS connections
DATABASE_URL="postgresql://user:pass@host:5432/db?sslmode=require"

# Connection encryption
DATABASE_URL="postgresql://user:pass@host:5432/db?sslmode=require&sslcert=client-cert.pem&sslkey=client-key.pem&sslrootcert=ca-cert.pem"
```

#### **Data Encryption**

- **At Rest**: RDS encryption enabled
- **In Transit**: SSL/TLS for all connections
- **Application Level**: Sensitive fields encrypted before storage

```python
# Example: Encrypt sensitive data
from cryptography.fernet import Fernet

def encrypt_voice_embedding(embedding_data: bytes) -> str:
    key = os.getenv('ENCRYPTION_KEY')
    f = Fernet(key)
    encrypted = f.encrypt(embedding_data)
    return base64.b64encode(encrypted).decode()
```

### Network Security

#### **VPC Configuration**

```bash
# Deploy Lambda in private VPC
aws lambda update-function-configuration \
  --function-name mira-backend \
  --vpc-config SubnetIds=subnet-12345,SecurityGroupIds=sg-12345

# RDS in private subnet
aws rds create-db-subnet-group \
  --db-subnet-group-name mira-db-subnet-group \
  --db-subnet-group-description "Private subnets for Mira DB" \
  --subnet-ids subnet-private-1 subnet-private-2
```

#### **Security Groups**

```bash
# Lambda security group (outbound only)
aws ec2 create-security-group \
  --group-name mira-lambda-sg \
  --description "Security group for Mira Lambda function"

# RDS security group (Lambda access only)
aws ec2 create-security-group \
  --group-name mira-rds-sg \
  --description "Security group for Mira RDS"

aws ec2 authorize-security-group-ingress \
  --group-id sg-rds-id \
  --protocol tcp \
  --port 5432 \
  --source-group sg-lambda-id
```

### API Security

#### **Input Validation**

```python
from pydantic import BaseModel, validator

class ConversationCreate(BaseModel):
    title: str
    participants: List[str]
    
    @validator('title')
    def validate_title(cls, v):
        if len(v) < 1 or len(v) > 255:
            raise ValueError('Title must be 1-255 characters')
        if not v.strip():
            raise ValueError('Title cannot be empty')
        return v.strip()
    
    @validator('participants')
    def validate_participants(cls, v):
        if not v:
            raise ValueError('At least one participant required')
        if len(v) > 50:
            raise ValueError('Maximum 50 participants allowed')
        return v
```

#### **Rate Limiting**

```python
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address

limiter = Limiter(key_func=get_remote_address)
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

@limiter.limit("100/hour")
@router.post("/{network_id}/conversations")
def create_conversation(request: Request, ...):
    # Rate limited endpoint
    pass
```

#### **CORS Configuration**

```python
# Production CORS settings
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "https://app.mira.ai",
        "https://dashboard.mira.ai"
    ],  # Specific origins only
    allow_credentials=True,
    allow_methods=["GET", "POST", "PUT", "DELETE"],
    allow_headers=["Authorization", "Content-Type"],
)
```

### Secrets Management

#### **AWS Secrets Manager**

```bash
# Store secrets in AWS Secrets Manager
aws secretsmanager create-secret \
  --name "mira/production/database" \
  --description "Database credentials for Mira production" \
  --secret-string '{"username":"mira_user","password":"SecurePassword123!"}'

aws secretsmanager create-secret \
  --name "mira/production/api-keys" \
  --description "API keys for Mira production" \
  --secret-string '{"gemini_key":"your_gemini_key_here"}'
```

#### **Lambda Environment Variables**

```python
import boto3
import json

def get_secret(secret_name):
    session = boto3.session.Session()
    client = session.client('secretsmanager', region_name='us-east-1')
    
    try:
        response = client.get_secret_value(SecretId=secret_name)
        secret = json.loads(response['SecretString'])
        return secret
    except Exception as e:
        raise e

# Usage
db_secrets = get_secret('mira/production/database')
DATABASE_URL = f"postgresql://{db_secrets['username']}:{db_secrets['password']}@{db_host}:5432/mira"
```

### Security Monitoring

#### **Security Alerts**

```bash
# Failed authentication attempts
aws logs create-log-group --log-group-name /aws/lambda/mira-backend/security

# Monitor for suspicious patterns
aws logs put-metric-filter \
  --log-group-name /aws/lambda/mira-backend \
  --filter-name SecurityEvents \
  --filter-pattern "[timestamp, request_id, level=ERROR, message=*authentication*]" \
  --metric-transformations \
    metricName=AuthenticationFailures,metricNamespace=Mira/Security,metricValue=1
```

#### **Security Scanning**

```bash
# Container vulnerability scanning
make security-scan

# Dependency vulnerability check
pip audit

# SAST (Static Application Security Testing)
bandit -r app/

# Infrastructure security
checkov -f docker/Dockerfile.lambda
```

### Compliance & Auditing

#### **Audit Logging**

```python
import logging

audit_logger = logging.getLogger('mira.audit')

def audit_log(action: str, resource: str, user_id: str = None, **kwargs):
    audit_logger.info(
        "Audit event",
        extra={
            "action": action,
            "resource": resource,
            "user_id": user_id,
            "timestamp": datetime.utcnow().isoformat(),
            "additional_data": kwargs
        }
    )

# Usage
audit_log("conversation.created", f"conversation:{conversation_id}", user_id=user_id)
audit_log("person.voice_updated", f"person:{person_id}", user_id=user_id)
```

#### **Data Retention**

```python
# Automated data cleanup
from datetime import datetime, timedelta

def cleanup_old_interactions():
    cutoff_date = datetime.utcnow() - timedelta(days=90)
    
    old_interactions = db.query(Interaction).filter(
        Interaction.created_at < cutoff_date
    ).delete()
    
    audit_log("data.cleanup", f"interactions:{old_interactions}")
    db.commit()
```

### Security Best Practices

1. **Principle of Least Privilege**: Grant minimal required permissions
2. **Defense in Depth**: Multiple layers of security controls
3. **Regular Updates**: Keep dependencies and runtime updated
4. **Security Testing**: Regular penetration testing and vulnerability assessments
5. **Incident Response**: Documented procedures for security incidents
6. **Backup & Recovery**: Regular backups with tested restore procedures

## 🤝 Contributing

We welcome contributions to the Mira Backend project! Please follow these guidelines for a smooth collaboration.

### Development Process

#### **1. Fork and Clone**

```bash
# Fork the repository on GitHub
# Clone your fork
git clone https://github.com/yourusername/backend.git
cd backend

# Add upstream remote
git remote add upstream https://github.com/mira-assistant/backend.git
```

#### **2. Create Feature Branch**

```bash
# Create and switch to feature branch
git checkout -b feature/your-feature-name

# Or for bug fixes
git checkout -b fix/your-bug-fix-name
```

#### **3. Development Setup**

```bash
# Setup development environment
cp .env.example .env
# Edit .env with your configuration

# Install dependencies
make install

# Run tests to ensure everything works
make test

# Start development server
make dev
```

#### **4. Make Changes**

Follow our coding standards:

- **Code Style**: Use Black for formatting (`make format`)
- **Linting**: Pass flake8 checks (`make lint`)
- **Type Hints**: Add type annotations for new code
- **Documentation**: Update docstrings and comments
- **Tests**: Add tests for new functionality

#### **5. Testing**

```bash
# Run all tests
make test

# Run specific test categories
make test-unit
make test-integration

# Check test coverage
make test-coverage

# Ensure tests pass before committing
```

#### **6. Commit Changes**

```bash
# Stage changes
git add .

# Commit with descriptive message
git commit -m "feat: add conversation search functionality

- Add search endpoint for conversations
- Implement full-text search with PostgreSQL
- Add pagination and filtering options
- Update API documentation

Closes #123"
```

#### **7. Submit Pull Request**

```bash
# Push feature branch
git push origin feature/your-feature-name

# Create pull request on GitHub
# Provide detailed description of changes
```

### Commit Message Format

We use conventional commits for consistent versioning and changelog generation:

```
<type>[optional scope]: <description>

[optional body]

[optional footer(s)]
```

#### **Types**

- `feat`: New feature
- `fix`: Bug fix
- `docs`: Documentation changes
- `style`: Code style changes (formatting, etc.)
- `refactor`: Code refactoring
- `test`: Adding or updating tests
- `chore`: Maintenance tasks

#### **Examples**

```bash
# Feature addition
git commit -m "feat(api): add conversation search endpoint"

# Bug fix
git commit -m "fix(db): resolve connection pool exhaustion"

# Documentation
git commit -m "docs: update deployment guide for new AWS regions"

# Breaking change
git commit -m "feat!: migrate to new authentication system

BREAKING CHANGE: The authentication flow has changed from network-based
to JWT tokens. Clients must update to use the new authentication headers."
```

### Code Review Process

#### **Pull Request Requirements**

Before submitting a PR, ensure:

- [ ] All tests pass (`make test`)
- [ ] Code is properly formatted (`make format`)
- [ ] Linting passes (`make lint`)
- [ ] Documentation is updated
- [ ] Breaking changes are documented
- [ ] Security implications are considered

#### **Review Checklist**

Reviewers will check:

- **Functionality**: Does the code work as intended?
- **Testing**: Are there adequate tests?
- **Performance**: Any performance implications?
- **Security**: Are there security considerations?
- **Documentation**: Is documentation updated?
- **Backwards Compatibility**: Any breaking changes?

#### **Review Process**

1. **Automated Checks**: CI/CD pipeline runs tests and checks
2. **Code Review**: Team members review the code
3. **Discussion**: Address any feedback or questions
4. **Approval**: Minimum 1 approval required
5. **Merge**: Squash and merge to main branch

### Development Guidelines

#### **API Development**

```python
# Use dependency injection
@router.get("/{network_id}/conversations/{conversation_id}")
def get_conversation(
    conversation_id: str = Path(..., description="The ID of the conversation"),
    network_id: str = Path(..., description="The ID of the network"),
    db: Session = Depends(get_db),
):
    """Get a conversation by ID."""
    # Implementation here

# Always validate inputs
from pydantic import BaseModel, validator

class ConversationUpdate(BaseModel):
    title: str
    
    @validator('title')
    def validate_title(cls, v):
        if not v.strip():
            raise ValueError('Title cannot be empty')
        return v.strip()
```

#### **Database Development**

```python
# Create migrations for schema changes
alembic revision --autogenerate -m "Add conversation search index"

# Use proper database sessions
from db import get_db

def get_conversations(db: Session = Depends(get_db)):
    try:
        conversations = db.query(Conversation).all()
        return conversations
    except Exception as e:
        db.rollback()
        raise
    finally:
        db.close()
```

#### **Testing Guidelines**

```python
# Unit tests
def test_create_conversation():
    # Test single function or method
    conversation = create_conversation(title="Test", participants=["user1"])
    assert conversation.title == "Test"

# Integration tests
def test_conversation_api_endpoint(client):
    # Test full API flow
    response = client.post("/api/v1/networks/test/conversations", 
                          json={"title": "Test Conversation"})
    assert response.status_code == 201
    assert response.json()["title"] == "Test Conversation"

# Use fixtures for common setup
@pytest.fixture
def sample_conversation():
    return Conversation(
        id=uuid4(),
        title="Sample Conversation",
        network_id=uuid4()
    )
```

### Issue Reporting

#### **Bug Reports**

When reporting bugs, include:

```markdown
## Bug Description
Brief description of the issue

## Steps to Reproduce
1. Step one
2. Step two
3. Step three

## Expected Behavior
What should happen

## Actual Behavior
What actually happens

## Environment
- OS: Ubuntu 20.04
- Python: 3.12.0
- Docker: 24.0.0
- Browser: Chrome 118.0

## Additional Context
Any additional information, logs, or screenshots
```

#### **Feature Requests**

For feature requests, provide:

```markdown
## Feature Description
Clear description of the proposed feature

## Use Case
Why is this feature needed? Who would use it?

## Proposed Solution
How should this feature work?

## Alternatives Considered
Any alternative approaches you've thought about

## Additional Context
Any additional information or examples
```

### Documentation

#### **Code Documentation**

```python
def process_audio_stream(
    stream_data: bytes, 
    quality_threshold: float = 0.8,
    noise_reduction: bool = True
) -> AudioProcessingResult:
    """Process incoming audio stream data.
    
    Args:
        stream_data: Raw audio data in bytes
        quality_threshold: Minimum quality score (0.0-1.0)
        noise_reduction: Whether to apply noise reduction
        
    Returns:
        AudioProcessingResult containing processed audio and metadata
        
    Raises:
        ValueError: If stream_data is empty or invalid
        ProcessingError: If audio processing fails
        
    Example:
        >>> stream_data = load_audio_file("sample.wav")
        >>> result = process_audio_stream(stream_data, quality_threshold=0.9)
        >>> print(f"Quality score: {result.quality_score}")
    """
    # Implementation here
```

#### **API Documentation**

All API endpoints should include:

- Clear descriptions
- Parameter definitions
- Response schemas
- Example requests/responses
- Error codes and descriptions

### Community Guidelines

#### **Code of Conduct**

- **Be Respectful**: Treat everyone with respect and kindness
- **Be Collaborative**: Work together towards common goals
- **Be Constructive**: Provide helpful feedback and suggestions
- **Be Patient**: Understand that everyone has different experience levels
- **Be Inclusive**: Welcome contributors from all backgrounds

#### **Communication Channels**

- **GitHub Issues**: Bug reports and feature requests
- **GitHub Discussions**: General questions and community discussions
- **Pull Requests**: Code review and technical discussions
- **Email**: team@mira.ai for private matters

#### **Recognition**

Contributors are recognized through:

- **Contributors file**: Listed in CONTRIBUTORS.md
- **Release notes**: Mentioned in changelog
- **GitHub**: Contributor badges and statistics

### Getting Help

If you need help:

1. **Check Documentation**: Review this README and API docs
2. **Search Issues**: Look for existing issues or discussions
3. **Ask Questions**: Create a GitHub discussion
4. **Join Community**: Participate in project discussions

Thank you for contributing to Mira Backend! 🚀

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

### MIT License

```
MIT License

Copyright (c) 2024 Mira Team

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```

### Third-Party Licenses

This project includes dependencies with their own licenses:

- **FastAPI**: MIT License
- **SQLAlchemy**: MIT License
- **Pydantic**: MIT License
- **PostgreSQL**: PostgreSQL License
- **Docker**: Apache License 2.0

For a complete list of dependencies and their licenses, see [requirements.txt](requirements.txt).

### Contributing License Agreement

By contributing to this project, you agree that your contributions will be licensed under the same MIT License that covers the project.

---

**Built with ❤️ by the Mira Team**

For questions or support, please contact us at [team@mira.ai](mailto:team@mira.ai) or create an issue on GitHub.
