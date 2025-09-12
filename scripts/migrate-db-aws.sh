#!/bin/bash

# Enterprise-grade database migration script for AWS RDS

set -euo pipefail

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Default values
STAGE="${STAGE:-dev}"
REGION="${REGION:-us-east-1}"

print_status() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

print_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Validate environment
validate_environment() {
    print_status "Validating environment..."

    if [ ! -f ".env" ]; then
        print_error ".env file not found"
        exit 1
    fi

    # Load environment variables
    set -a
    source .env
    set +a

    if [ -z "${DATABASE_URL:-}" ]; then
        print_error "DATABASE_URL not found in .env file"
        exit 1
    fi

    print_success "Environment validation passed"
}

# Run database migrations
run_migrations() {
    print_status "Running database migrations..."

    # Set environment variable for Alembic
    export DATABASE_URL

    # Run migrations
    alembic upgrade head

    print_success "Database migrations completed"
}

# Main execution
main() {
    print_status "Starting database migration for $STAGE environment"

    validate_environment
    run_migrations

    print_success "Database migration completed successfully!"
}

# Run main function
main "$@"
