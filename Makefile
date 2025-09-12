.PHONY: help install test lint format clean docker-build docker-run dev

help: ## Show this help message
	@echo 'Usage: make [target]'
	@echo ''
	@echo 'Targets:'
	@awk 'BEGIN {FS = ":.*?## "} /^[a-zA-Z_-]+:.*?## / {printf "  %-15s %s\n", $$1, $$2}' $(MAKEFILE_LIST)

install: ## Install dependencies
	pip install -r requirements.txt
	python -m spacy download en_core_web_sm

test: ## Run tests (native)
	pytest

test-docker: ## Run tests in Docker (matches CI environment)
	docker run --rm -v $(PWD):/app -w /app -e DATABASE_URL=sqlite:///./test.db mira-backend sh -c "mkdir -p /tmp/test-output && pytest --cov=app --cov-report=xml:/tmp/test-output/coverage.xml --cov-report=html:/tmp/test-output/htmlcov --junitxml=/tmp/test-output/test-results.xml && cp -r /tmp/test-output/* /app/ 2>/dev/null || true"

test-cov: ## Run tests with coverage
	pytest --cov=app --cov-report=html --cov-report=term

lint: ## Run linting (native)
	flake8 . --count --select=E9,F63,F7,F82 --show-source --statistics
	flake8 . --count --exit-zero --max-complexity=10 --max-line-length=88 --statistics
	black --check app/
	isort --check-only app/

lint-docker: ## Run linting in Docker (matches CI environment)
	docker run --rm -v $(PWD):/app -w /app mira-backend flake8 app/ --count --select=E9,F63,F7,F82 --show-source --statistics
	docker run --rm -v $(PWD):/app -w /app mira-backend black --check app/
	docker run --rm -v $(PWD):/app -w /app mira-backend isort --check-only app/

format: ## Format code
	black app/
	isort app/

clean: ## Clean up temporary files
	find . -type f -name "*.pyc" -delete
	find . -type d -name "__pycache__" -delete
	find . -type d -name "*.egg-info" -exec rm -rf {} +
	rm -rf .pytest_cache/
	./scripts/cleanup-coverage.sh

clean-coverage: ## Clean up only coverage files
	./scripts/cleanup-coverage.sh

docker-build: ## Build Docker image
	docker build -f docker/Dockerfile.dev -t mira-backend .

docker-run: ## Run Docker container
	docker run -p 8000:8000 mira-backend

docker-compose-up: ## Start dedvelopment environment with Docker Compose
	cd docker && docker-compose -f docker-compose.dev.yml up --build

docker-compose-down: ## Stop development environment
	cd docker && docker-compose -f docker-compose.dev.yml down

docker-compose-logs: ## View logs from Docker Compose
	cd docker && docker-compose -f docker-compose.dev.yml logs -f

dev: ## Run development server
	uvicorn app.main:app --reload --host 0.0.0.0 --port 8000

db-init: ## Initialize database
	python scripts/init_db.py

db-migrate: ## Run database migrations
	alembic upgrade head

db-migrate-aws: ## Run database migrations on AWS RDS
	./scripts/migrate-db.sh

deploy: ## Deploy to AWS Lambda (enterprise)
	./scripts/deployment/deploy-infrastructure.sh $(ENV) && \
	./scripts/deployment/deploy-application.sh $(ENV) && \
	./scripts/deployment/migrate-database.sh $(ENV) && \
	./scripts/deployment/health-check.sh $(ENV)

deploy-infra: ## Deploy infrastructure only
	./scripts/deployment/deploy-infrastructure.sh $(ENV)

deploy-app: ## Deploy application only
	./scripts/deployment/deploy-application.sh $(ENV)

migrate-db: ## Run database migrations
	./scripts/deployment/migrate-database.sh $(ENV)

health-check: ## Run health checks
	./scripts/deployment/health-check.sh $(ENV)

db-reset: ## Reset database
	python -c "from app.db.init_db import reset_db; reset_db()"
