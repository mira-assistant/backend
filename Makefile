.PHONY: help
help:
	@echo "Usage: make <target>"
	@echo "Targets:"
	@echo "  dev - Run development server"
	@echo "  lambda - Run deployment server"
	@echo "  dev-build - Build development container"
	@echo "  lambda-build - Build deployment container"
	@echo "  dev-run - Start development container"
	@echo "  lambda-run - Start deployment container"
	@echo "  stop-containers - Stop all running containers"
	@echo "  test - Run tests"
	@echo "  lint - Run linting tools"
	@echo "  format - Format code with ruff"
	@echo "  clean - Clean up containers and images"


.PHONY: dev
dev:
	make dev-build
	make dev-run

.PHONY: lambda
lambda:
	make lambda-build
	make lambda-run

dev-build:
	docker build -t mira-api:dev -f docker/Dockerfile.dev .
lambda-build:
	docker build -t mira-api:lambda -f docker/Dockerfile.lambda .

dev-run:
	docker run --rm -it --env-file .env -v $(PWD)/app:/app -p 8000:8000 mira-api:dev
lambda-run:
	docker run --rm -it --env-file .env -p 9000:8080 mira-api:lambda

.PHONY: stop
stop-containera:
	@echo "Stopping all running containers..."
	@docker stop $$(docker ps -q) 2>/dev/null || echo "No containers to stop"


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