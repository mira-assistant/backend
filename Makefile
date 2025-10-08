.PHONY: help
help:
	@echo "Usage: make <target>"
	@echo "Targets:"
	@echo "  dev           - Run development server"
	@echo "  lambda        - Run deployment server"
	@echo "  prod          - Run production server"
	@echo "  dev-build     - Build development container"
	@echo "  lambda-build  - Build deployment container"
	@echo "  prod-build    - Build production container"
	@echo "  dev-run       - Start development container"
	@echo "  lambda-run    - Start deployment container"
	@echo "  prod-run      - Start production container"
	@echo "  test          - Run tests"
	@echo "  lint          - Run linting tools"
	@echo "  format        - Format code with ruff"
	@echo "  clean         - Clean up containers and images"


.PHONY: dev
dev:
	make dev-build
	make dev-run

.PHONY: lambda
lambda:
	make lambda-build
	make lambda-run

.PHONY
prod:
	make prod-build
	make prod-run

dev-build:
	docker build -t mira-api:dev -f docker/Dockerfile.dev .
lambda-build:
	docker build -t mira-api:lambda -f docker/Dockerfile.lambda .
prod-build:
	docker build -t mira-api:prod -f docker/Dockerfile.prod .

dev-run:
	docker run --rm -it --env-file .env -v $(PWD)/app:/app -p 8000:8000 mira-api:dev
lambda-run:
	docker run --rm -it --env-file .env -p 9000:8080 mira-api:lambda
prod-run:
	docker run -it -p 80:80 mira-api:prod

.PHONY: test
test:
	@echo "Running tests..."

.PHONY: lint
lint:
	docker run --rm -v $(PWD)/app:/app -w /app mira-api:dev ruff check .

.PHONY: format
format:
	docker run --rm -v $(PWD)/app:/app -w /app mira-api:dev ruff check --fix .
	docker run --rm -v $(PWD)/app:/app -w /app mira-api:dev ruff format .

.PHONY: clean
clean:
	@echo "Cleaning up containers and images..."
	@docker stop $$(docker ps -q) 2>/dev/null || true
	@docker rm $$(docker ps -aq) 2>/dev/null || true
	@docker rmi mira-api:dev 2>/dev/null || true
	@echo "Cleanup complete!"