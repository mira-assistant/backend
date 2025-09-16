.PHONY: help
help:
	@echo "Usage: make <target>"
	@echo "Targets:"
	@echo "  dev - Run development server"
	@echo "  lambda - Run deployment server"
	@echo "  build-container-dev - Build development container"
	@echo "  build-container-lambda - Build deployment container"
	@echo "  run-container-dev - Start development container"
	@echo "  run-container-lambda - Start deployment container"
	@echo "  stop-containers - Stop all running containers"
	@echo "  test - Run tests"
	@echo "  lint - Run linting tools"
	@echo "  format - Format code with isort and black"
	@echo "  clean - Clean up containers and images"


.PHONY: dev
dev:
	make build-container-dev
	make run-container-dev

.PHONY: lambda
lambda:
	make build-container-lambda
	make run-container-lambda

.PHONY: build
build-container-dev:
	docker build -t mira-api:dev -f docker/Dockerfile.dev .

build-container-lambda:
	docker build -t mira-api:lambda -f docker/Dockerfile.lambda .

.PHONY: run
run-container-dev:
	docker run --rm -it -v $(PWD)/app:/app -w /app -p 8000:8000 mira-api:dev

run-container-lambda:
	docker run --rm -it -v $(PWD)/app:/app -w /app -p 8000:8000 mira-api:lambda uvicorn main:app --host 0.0.0.0 --port 8000

.PHONY: stop
stop-containera:
	@echo "Stopping all running containers..."
	@docker stop $$(docker ps -q) 2>/dev/null || echo "No containers to stop"


.PHONY: test
test:
	@echo "Running tests..."

.PHONY: lint
lint:
	docker run --rm -v $(PWD)/app:/app -w /app mira-api:dev flake8 --count --select=E9,F63,F7,F82 --show-source --statistics
	docker run --rm -v $(PWD)/app:/app -w /app mira-api:dev black --check --diff .

.PHONY: format
format:
	docker run --rm -v $(PWD)/app:/app -w /app mira-api:dev isort .
	docker run --rm -v $(PWD)/app:/app -w /app mira-api:dev black .

.PHONY: clean
clean:
	@echo "Cleaning up containers and images..."
	@docker stop $$(docker ps -q) 2>/dev/null || true
	@docker rm $$(docker ps -aq) 2>/dev/null || true
	@docker rmi mira-api:dev 2>/dev/null || true
	@echo "Cleanup complete!"