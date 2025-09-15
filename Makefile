.PHONY: help
help:
	@echo "Usage: make <target>"
	@echo "Targets:"
	@echo "  dev - Run development server"
	@echo "  build-container - Build development container"
	@echo "  run-container - Run development container"


.PHONY: dev
dev:
	docker build -t mira-backend:dev -f docker/Dockerfile.dev .
	docker run --rm -it -v $(PWD)/app:/app -w /app -p 8000:8000 mira-backend:dev

.PHONY: build
build-container:
	docker build -t mira-backend:dev -f docker/Dockerfile.dev .

.PHONY: run
run-container:
	docker run --rm -it -v $(PWD)/app:/app -w /app -p 8000:8000 mira-backend:dev
