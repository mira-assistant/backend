# Mira Backend

A FastAPI-based backend service for the Mira AI assistant, deployed on AWS Lambda using container images.

## 🚀 Quick Start

### Prerequisites
- Python 3.12+
- Docker
- AWS CLI configured
- PostgreSQL database (RDS recommended)

### Local Development

1. **Clone and setup:**
   ```bash
   git clone <repository-url>
   cd backend
   cp .env.example .env
   # Edit .env with your configuration
   ```

2. **Install dependencies:**
   ```bash
   make install
   ```

3. **Run development server:**
   ```bash
   make dev
   ```

### Docker Development

1. **Build and run with Docker:**
   ```bash
   make docker-build
   make docker-run
   ```

2. **Or use Docker Compose:**
   ```bash
   make docker-compose-up
   ```

## 🐳 Container Deployment

This project uses **container-based AWS Lambda deployment** to handle large ML dependencies.

### Deploy to AWS

1. **Set up GitHub secrets:**
   - `AWS_ACCESS_KEY_ID`
   - `AWS_SECRET_ACCESS_KEY`
   - `DATABASE_URL`
   - `GEMINI_API_KEY`

2. **Deploy via GitHub Actions:**
   - Push to main branch triggers automatic deployment

3. **Deploy locally:**
   ```bash
   make deploy ENV=dev
   ```

### Test Container Deployment

```bash
make docker-test
```

## 📁 Project Structure

```
├── app/                    # FastAPI application
│   ├── api/               # API routes (v1, v2)
│   ├── core/              # Configuration and utilities
│   ├── db/                # Database models and session
│   ├── models/            # SQLAlchemy models
│   ├── services/          # Business logic
│   └── tests/             # Test files
├── docker/                # Docker configurations
│   ├── Dockerfile.dev     # Development container
│   └── Dockerfile.lambda  # Lambda deployment container
├── scripts/               # Deployment and utility scripts
│   └── deployment/        # AWS deployment scripts
└── alembic/              # Database migrations
```

## 🛠️ Available Commands

### Development
- `make dev` - Run development server
- `make test` - Run tests
- `make lint` - Run linting
- `make format` - Format code

### Docker
- `make docker-build` - Build development image
- `make docker-build-lambda` - Build Lambda image
- `make docker-run` - Run development container
- `make docker-test` - Test container deployment

### Database
- `make db-migrate` - Run migrations
- `make db-migrate-aws` - Run migrations on AWS RDS

### Deployment
- `make deploy` - Full deployment
- `make deploy-infra` - Deploy infrastructure only
- `make deploy-app` - Deploy application only

## 🔧 Configuration

### Environment Variables

Required in `.env`:
- `DATABASE_URL` - PostgreSQL connection string
- `GEMINI_API_KEY` - Google Gemini API key

### AWS Configuration

The deployment creates:
- Lambda function with container image
- ECR repository for container storage
- API Gateway for HTTP access
- IAM roles with necessary permissions

## 📊 Monitoring

- **CloudWatch Logs**: Lambda function logs
- **API Gateway**: Request/response metrics
- **RDS**: Database performance metrics

## 🤝 Contributing

1. Create feature branch
2. Make changes
3. Run tests: `make test`
4. Run linting: `make lint`
5. Submit pull request

## 📝 License

[Your License Here]
