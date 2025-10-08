.PHONY: help up down build lint format test clean db

help:
	@echo "Usage: make <target>"
	@echo "Targets:"
	@echo "  up      - Start containers"
	@echo "  down    - Stop containers"
	@echo "  build   - Build the API container"
	@echo "  lint    - Run linting tools"
	@echo "  format  - Format code with ruff"
	@echo "  test    - Run tests"
	@echo "  clean   - Remove containers and images"

up:
	docker-compose up -d --build

down:
	docker-compose down

build:
	docker-compose build api

lint:
	docker-compose run --rm api ruff check /app

format:
	docker-compose run --rm api ruff check --fix /app
	docker-compose run --rm api ruff format /app

test:
	docker-compose run --rm api pytest /app

clean:
	docker-compose down -v
	docker rmi mira-api:dev || true

db-upgrade:
	docker-compose run --rm api alembic upgrade head