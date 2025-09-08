.PHONY: help install test lint format clean docker-build docker-run dev

help: ## Show this help message
	@echo 'Usage: make [target]'
	@echo ''
	@echo 'Targets:'
	@awk 'BEGIN {FS = ":.*?## "} /^[a-zA-Z_-]+:.*?## / {printf "  %-15s %s\n", $$1, $$2}' $(MAKEFILE_LIST)

install: ## Install dependencies
	pip install -r requirements.txt
	python -m spacy download en_core_web_sm

test: ## Run tests
	pytest

test-cov: ## Run tests with coverage
	pytest --cov=app --cov-report=html --cov-report=term

lint: ## Run linting
	flake8 app/ --count --select=E9,F63,F7,F82 --show-source --statistics
	black --check app/
	isort --check-only app/

format: ## Format code
	black app/
	isort app/

clean: ## Clean up temporary files
	find . -type f -name "*.pyc" -delete
	find . -type d -name "__pycache__" -delete
	find . -type d -name "*.egg-info" -exec rm -rf {} +
	rm -rf .pytest_cache/
	rm -rf htmlcov/
	rm -rf .coverage
	rm -rf .serverless/

docker-build: ## Build Docker image
	docker build -t mira-backend .

docker-run: ## Run Docker container
	docker run -p 8000:8000 mira-backend

dev: ## Run development server
	uvicorn app.main:app --reload --host 0.0.0.0 --port 8000

db-init: ## Initialize database
	python scripts/init_db.py

db-migrate: ## Run database migrations
	alembic upgrade head

db-migrate-aws: ## Run database migrations on AWS RDS
	./scripts/migrate-db.sh

deploy: ## Deploy to AWS Lambda
	./scripts/deploy.sh

setup-aws: ## Setup AWS Lambda environment
	./scripts/setup-lambda.sh

db-reset: ## Reset database
	python -c "from app.db.init_db import reset_db; reset_db()"
