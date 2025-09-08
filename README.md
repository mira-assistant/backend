# 🎤 Mira Backend

A real-time audio processing and AI conversation backend built with FastAPI, featuring speaker diarization, wake word detection, and multi-LLM support.

## ✨ Features

- **Real-time Audio Processing** - Live speech recognition with Whisper
- **Speaker Diarization** - Identify and track multiple speakers
- **Wake Word Detection** - Customizable wake words for voice activation
- **Multi-LLM Support** - Gemini, OpenAI, Anthropic, LM Studio, and AWS Bedrock
- **WebSocket Streaming** - Real-time audio streaming and responses
- **Database Integration** - SQLite (local) and PostgreSQL (production)
- **AWS Lambda Ready** - Serverless deployment with RDS

## 🚀 Quick Start

### Local Development

```bash
# 1. Clone and setup
git clone <your-repo>
cd mira-backend

# 2. Install dependencies
make install

# 3. Run tests
make test

# 4. Start development server
make dev
```

### AWS Lambda Deployment

```bash
# 1. Setup AWS environment
make setup-aws

# 2. Deploy to AWS Lambda
make deploy
```

## 📁 Project Structure

```
mira-backend/
├── app/                    # Main application code
│   ├── api/               # FastAPI routers (v1 & v2)
│   ├── core/              # Configuration and constants
│   ├── db/                # Database models and session
│   ├── models/            # SQLAlchemy models
│   ├── services/          # Business logic services
│   └── tests/             # Test suite
├── scripts/               # Deployment and migration scripts
├── docs/                  # Documentation
└── serverless.yml         # AWS Lambda configuration
```

## 🛠️ Development

### Available Commands

```bash
make help          # Show all available commands
make test          # Run tests with coverage
make lint          # Run linting
make format        # Format code
make dev           # Start development server
make clean         # Clean up temporary files
```

### Configuration

- **Constants**: `app/core/constants.py` - Business logic constants
- **Settings**: `app/core/config.py` - Environment-dependent settings
- **Environment**: `.env` - Local environment variables

## 📚 Documentation

- [Deployment Guide](docs/deployment.md) - AWS Lambda and RDS setup
- [Database Guide](docs/database.md) - Database configuration and migrations
- [API Reference](docs/api.md) - API endpoints and usage

## 🔧 Configuration

### Environment Variables

Create a `.env` file with your credentials:

```bash
# LLM Backend (choose one)
GEMINI_API_KEY=your_gemini_key
OPENAI_API_KEY=your_openai_key
LLM_BACKEND=gemini

# Database
DATABASE_URL=sqlite:///./mira.db
```

### Supported LLM Backends

- **Gemini** - Google's Gemini API
- **OpenAI** - GPT models
- **Anthropic** - Claude models
- **LM Studio** - Local models
- **AWS Bedrock** - Amazon's managed models

## 🧪 Testing

```bash
# Run all tests
make test

# Run specific test file
pytest app/tests/unit/test_services.py -v

# Run with coverage
make test-cov
```

## 🚀 Deployment

### AWS Lambda (Recommended)

```bash
# 1. Setup AWS environment
./scripts/setup-lambda.sh

# 2. Deploy to AWS
./scripts/deploy.sh

# 3. Run database migrations
./scripts/migrate-db.sh
```

### Docker (Alternative)

```bash
# Build and run with Docker
make docker-build
make docker-run
```

## 📖 API Usage

### WebSocket Connection

```javascript
const ws = new WebSocket('ws://localhost:8000/ws/{network_id}');
ws.send(audioData); // Send PCM audio data
```

### REST Endpoints

- `POST /api/v1/interactions` - Process audio interactions
- `GET /api/v1/persons` - List speakers
- `GET /api/v1/conversations` - Get conversation history

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Run tests: `make test`
5. Submit a pull request

## 📄 License

[Your License Here]

---

**Need help?** Check the [documentation](docs/) or open an issue.
