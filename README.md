# Mira Backend API

A modern FastAPI-based backend for the Mira AI Assistant, following industry best practices and standards.

## 🏗️ Architecture

This backend follows a clean, modular architecture with clear separation of concerns:

```
mira-backend/
├── app/                        # Main application package
│   ├── __init__.py
│   ├── main.py                  # FastAPI app entrypoint
│   ├── api/                     # API layer
│   │   ├── __init__.py
│   │   ├── v1/                  # Versioned API
│   │   │   ├── __init__.py
│   │   │   ├── auth.py          # Authentication endpoints
│   │   │   ├── tasks.py         # Task management endpoints
│   │   │   └── assistant.py     # AI assistant endpoints
│   │   └── deps.py              # Dependencies (Depends)
│   ├── core/                    # Core settings, config, utils
│   │   ├── __init__.py
│   │   ├── config.py            # App settings (pydantic BaseSettings)
│   │   ├── security.py          # Auth utils (JWT, hashing)
│   │   └── logging.py           # Centralized logging
│   ├── models/                  # SQLAlchemy models
│   │   ├── __init__.py
│   │   ├── user.py
│   │   ├── person.py
│   │   ├── interaction.py
│   │   ├── conversation.py
│   │   └── action.py
│   ├── schemas/                 # Pydantic schemas
│   │   ├── __init__.py
│   │   ├── user.py
│   │   ├── person.py
│   │   ├── interaction.py
│   │   ├── conversation.py
│   │   └── action.py
│   ├── services/                # Business logic & integrations
│   │   ├── __init__.py
│   │   ├── ai_engine.py         # AI model integration
│   │   ├── command_service.py   # Command processing
│   │   ├── inference_service.py # Action inference
│   │   └── context_service.py   # Context processing
│   ├── db/                      # Database layer
│   │   ├── __init__.py
│   │   ├── base.py              # Base SQLAlchemy metadata
│   │   ├── session.py           # Session local, engine
│   │   └── init_db.py           # Database initialization
│   ├── workers/                 # Background tasks
│   │   └── __init__.py
│   └── tests/                   # Test suite
│       ├── __init__.py
│       ├── conftest.py
│       ├── test_auth.py
│       ├── test_tasks.py
│       └── test_assistant.py
├── alembic/                     # Database migrations
│   ├── versions/
│   ├── env.py
│   └── script.py.mako
├── scripts/                     # DevOps, setup scripts
│   ├── run_server.sh
│   └── init_data.py
├── .env                         # Environment variables
├── alembic.ini                  # Alembic config
├── requirements.txt             # Python dependencies
├── Dockerfile
├── docker-compose.yml
└── README.md
```

## 🚀 Quick Start

### Prerequisites

- Python 3.11+
- LM Studio (for AI model inference)
- Git

### Installation

1. **Clone the repository**
   ```bash
   git clone <repository-url>
   cd mira-backend
   ```

2. **Create virtual environment**
   ```bash
   python3 -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Set up environment variables**
   ```bash
   cp env.example .env
   # Edit .env with your configuration
   ```

5. **Initialize database**
   ```bash
   alembic upgrade head
   python scripts/init_data.py
   ```

6. **Start the server**
   ```bash
   python -m uvicorn app.main:app --reload
   ```

The API will be available at `http://localhost:8000`

### Using Docker

1. **Build and run with Docker Compose**
   ```bash
   docker-compose up --build
   ```

2. **Or build and run manually**
   ```bash
   docker build -t mira-backend .
   docker run -p 8000:8000 mira-backend
   ```

## 📚 API Documentation

Once the server is running, you can access:

- **Interactive API docs**: http://localhost:8000/docs
- **ReDoc documentation**: http://localhost:8000/redoc
- **OpenAPI schema**: http://localhost:8000/openapi.json

### Key Endpoints

#### Authentication
- `POST /api/v1/auth/register` - Register new user
- `POST /api/v1/auth/login` - Login user
- `GET /api/v1/auth/me` - Get current user info

#### Assistant
- `POST /api/v1/assistant/interactions/register` - Register audio interaction
- `GET /api/v1/assistant/interactions/{id}` - Get interaction
- `POST /api/v1/assistant/interactions/{id}/inference` - Run inference
- `DELETE /api/v1/assistant/interactions/{id}` - Delete interaction

#### Tasks
- `POST /api/v1/tasks/` - Create new task/action
- `GET /api/v1/tasks/` - Get user tasks
- `GET /api/v1/tasks/{id}` - Get specific task
- `PUT /api/v1/tasks/{id}` - Update task
- `DELETE /api/v1/tasks/{id}` - Delete task

## 🗄️ Database Management

### Migrations

The project uses Alembic for database migrations:

```bash
# Create a new migration
alembic revision --autogenerate -m "Description of changes"

# Apply migrations
alembic upgrade head

# Rollback migration
alembic downgrade -1
```

### Database Schema

The database includes the following main entities:

- **Users**: Authentication and user management
- **Persons**: Speaker recognition and voice profiles
- **Interactions**: Conversation interactions with NLP features
- **Conversations**: Conversation sessions and context
- **Actions**: User actions and tasks

## 🧪 Testing

Run the test suite:

```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=app

# Run specific test file
pytest app/tests/test_auth.py
```

## 🔧 Configuration

Configuration is managed through environment variables and Pydantic settings:

### Environment Variables

| Variable | Description | Default |
|----------|-------------|---------|
| `DATABASE_URL` | Database connection string | `sqlite:///./mira.db` |
| `LM_STUDIO_URL` | LM Studio server URL | `http://localhost:1234/v1` |
| `SECRET_KEY` | JWT secret key | `your-secret-key-here` |
| `DEBUG` | Debug mode | `false` |
| `LOG_LEVEL` | Logging level | `INFO` |

### LM Studio Integration

The backend integrates with LM Studio for AI model inference:

1. Start LM Studio server
2. Load your preferred models
3. Configure `LM_STUDIO_URL` in environment variables

## 🏭 Production Deployment

### Using Docker

1. **Build production image**
   ```bash
   docker build -t mira-backend:latest .
   ```

2. **Run with production settings**
   ```bash
   docker run -d \
     --name mira-backend \
     -p 8000:8000 \
     -e DATABASE_URL=postgresql://user:pass@host:port/db \
     -e SECRET_KEY=your-production-secret \
     -e DEBUG=false \
     mira-backend:latest
   ```

### Environment Setup

For production, ensure you have:

- Strong `SECRET_KEY`
- Production database (PostgreSQL recommended)
- Proper CORS configuration
- SSL/TLS termination
- Monitoring and logging

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests for new functionality
5. Ensure all tests pass
6. Submit a pull request

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🆘 Support

For support and questions:

- Create an issue in the repository
- Check the API documentation at `/docs`
- Review the test files for usage examples

## 🔄 Migration from Legacy Code

This restructured version maintains backward compatibility where possible. Key changes:

- **Modular architecture**: Code is now organized into logical modules
- **Type safety**: Full type hints and Pydantic schemas
- **Database migrations**: Alembic for schema management
- **Authentication**: JWT-based authentication system
- **Testing**: Comprehensive test suite
- **Documentation**: Auto-generated API documentation

The legacy processor files have been refactored into the services layer while maintaining the same functionality.


