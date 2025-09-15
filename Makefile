.PHONY: help
help:
	@echo "Usage: make <target>"
	@echo "Targets:"
	@echo "  dev - Run development server"
	@echo "  build-container - Build development container"
	@echo "  run-container - Run development container"
	@echo "  stop-container - Stop all running containers"
	@echo "  test - Run tests"
	@echo "  lint - Run linting tools"
	@echo "  clean - Clean up containers and images"


.PHONY: dev
dev:
	docker build -t mira-api:dev -f docker/Dockerfile.dev .
	docker run --rm -it -v $(PWD)/app:/app -w /app -p 8000:8000 mira-api:dev

.PHONY: build
build-container:
	docker build -t mira-api:dev -f docker/Dockerfile.dev .

.PHONY: run
run-container:
	docker run --rm -it -v $(PWD)/app:/app -w /app -p 8000:8000 mira-api:dev

.PHONY: stop
stop-container:
	@echo "Stopping all running containers..."
	@docker stop $$(docker ps -q) 2>/dev/null || echo "No containers to stop"

.PHONY: test
test:
	@echo "Running tests..."

.PHONY: lint
lint:
	docker run --rm -v $(PWD)/app:/app -w /app mira-api:dev flake8 --count --select=E9,F63,F7,F82 --show-source --statistics
	docker run --rm -v $(PWD)/app:/app -w /app mira-api:dev black --check --diff .
	docker run --rm -v $(PWD)/app:/app -w /app mira-api:dev isort --check-only --diff .

.PHONY: clean
clean:
	@echo "Cleaning up containers and images..."
	@docker stop $$(docker ps -q) 2>/dev/null || true
	@docker rm $$(docker ps -aq) 2>/dev/null || true
	@docker rmi mira-api:dev 2>/dev/null || true
	@echo "Cleanup complete!"