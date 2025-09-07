# CI/CD Pipeline Documentation

## Overview

This document describes the CI/CD pipeline setup for the Mira Backend project.

## GitHub Actions Workflow

The CI/CD pipeline is defined in `.github/workflows/backend-ci.yml` and includes:

### Test Job
- **Triggers**: Push to main/develop branches, PRs to main/develop
- **Matrix Strategy**: Tests on Python 3.10 and 3.11
- **Steps**:
  1. Checkout code
  2. Set up Python environment
  3. Cache pip dependencies
  4. Install dependencies
  5. Install spaCy model (`en_core_web_sm`)
  6. Run linting (flake8, black, isort)
  7. Run tests with coverage
  8. Upload coverage to Codecov

### Build Job
- **Triggers**: Only on main branch after tests pass
- **Steps**:
  1. Checkout code
  2. Set up Python environment
  3. Install dependencies
  4. Build Docker image
  5. Test Docker image

## Local Development

### Prerequisites
- Python 3.10+
- pip
- Docker (optional)

### Setup
```bash
# Install dependencies
make install

# Initialize database
make db-init

# Run tests
make test

# Run development server
make dev
```

### Available Commands
- `make help` - Show all available commands
- `make install` - Install dependencies
- `make test` - Run tests
- `make test-cov` - Run tests with coverage
- `make lint` - Run linting
- `make format` - Format code
- `make clean` - Clean temporary files
- `make docker-build` - Build Docker image
- `make docker-run` - Run Docker container
- `make dev` - Run development server
- `make db-init` - Initialize database
- `make db-migrate` - Run database migrations
- `make db-reset` - Reset database

## Docker Support

### Dockerfile
The `Dockerfile` creates a production-ready image with:
- Python 3.10 slim base image
- System dependencies for ML libraries
- Non-root user for security
- Health check endpoint
- Optimized layer caching

### Docker Compose
The `docker-compose.yml` provides a complete development environment with:
- Backend service
- PostgreSQL database
- Redis cache
- Volume persistence

### Usage
```bash
# Start all services
docker-compose up -d

# View logs
docker-compose logs -f backend

# Stop services
docker-compose down
```

## Code Quality

### Linting
- **flake8**: Python style guide enforcement
- **black**: Code formatting
- **isort**: Import sorting

### Testing
- **pytest**: Test framework
- **coverage**: Code coverage reporting
- **httpx**: HTTP client for API testing

### Pre-commit Hooks (Recommended)
Create `.pre-commit-config.yaml`:
```yaml
repos:
  - repo: https://github.com/psf/black
    rev: 23.3.0
    hooks:
      - id: black
  - repo: https://github.com/pycqa/isort
    rev: 5.12.0
    hooks:
      - id: isort
  - repo: https://github.com/pycqa/flake8
    rev: 6.0.0
    hooks:
      - id: flake8
```

## Deployment

### Production Considerations
1. Use environment variables for configuration
2. Set up proper logging
3. Configure health checks
4. Use a reverse proxy (nginx)
5. Set up monitoring and alerting
6. Use secrets management for sensitive data

### Environment Variables
- `DATABASE_URL`: Database connection string
- `DEBUG`: Enable debug mode (false in production)
- `CORS_ORIGINS`: Allowed CORS origins
- `API_KEY`: API authentication key

## Monitoring

### Health Checks
- Endpoint: `GET /`
- Docker health check included
- Returns system status and version

### Logging
- Structured logging with custom formatter
- Different log levels for different environments
- Request/response logging

## Troubleshooting

### Common Issues
1. **spaCy model not found**: Run `python -m spacy download en_core_web_sm`
2. **Database connection issues**: Check DATABASE_URL environment variable
3. **Import errors**: Ensure all dependencies are installed
4. **Test failures**: Check that all required models are downloaded

### Debug Mode
Set `DEBUG=true` to enable:
- Detailed error messages
- SQL query logging
- Development-specific configurations
