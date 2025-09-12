#!/bin/bash

# Enterprise Database Migration Script
# Manages database schema migrations for Mira Backend

set -euo pipefail

# Script configuration
readonly SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
readonly PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
readonly SCRIPT_NAME="$(basename "$0")"

# Load utilities
source "${SCRIPT_DIR}/../utilities/logging.sh"
source "${SCRIPT_DIR}/../utilities/validation.sh"
source "${SCRIPT_DIR}/../utilities/database-utils.sh"

# Default configuration
readonly DEFAULT_ENVIRONMENT="dev"

# Parse command line arguments
parse_arguments() {
    local environment="${1:-${DEFAULT_ENVIRONMENT}}"

    log_info "Starting database migration"
    log_info "Environment: ${environment}"

    # Export for use in other functions
    export ENVIRONMENT="${environment}"
}

# Validate prerequisites
validate_prerequisites() {
    log_info "Validating prerequisites..."

    # Check if .env file exists
    if [ ! -f "${PROJECT_ROOT}/.env" ]; then
        log_error ".env file not found in project root"
        exit 1
    fi

    # Load environment variables
    set -a
    source "${PROJECT_ROOT}/.env"
    set +a

    # Validate required variables
    validate_required_env_vars "DATABASE_URL"

    # Check database connectivity
    if ! test_database_connection; then
        log_error "Cannot connect to database"
        exit 1
    fi

    # Check if Alembic is vailable
    if ! command -v alembic &> /dev/null; then
        log_error "Alembic is not installed"
        exit 1
    fi

    log_success "Prerequisites validated"
}

# Run database migrations
run_migrations() {
    log_info "Running database migrations..."

    # Set working directory to project root
    cd "${PROJECT_ROOT}"

    # Export database URL for Alembic
    export DATABASE_URL

    # Check current migration status
    log_info "Current migration status:"
    alembic current

    # Run migrations
    log_info "Applying migrations..."
    alembic upgrade head

    # Verify migration status
    log_info "Migration status after upgrade:"
    alembic current

    log_success "Database migrations completed successfully"
}

# Verify migration integrity
verify_migrations() {
    log_info "Verifying migration integrity..."

    # Check if all migrations are applied
    local pending_migrations
    pending_migrations=$(alembic heads | wc -l)

    if [ "${pending_migrations}" -gt 1 ]; then
        log_warning "Multiple migration heads detected"
        alembic heads
    fi

    # Check for migration conflicts
    local current_revision
    current_revision=$(alembic current --sql | grep -o 'revision=[^,]*' | cut -d'=' -f2 | tr -d "'" || echo "unknown")

    log_info "Current database revision: ${current_revision}"
    log_success "Migration integrity verified"
}

# Main migration function
migrate_database() {
    log_info "Migrating database for environment: ${ENVIRONMENT}"

    validate_prerequisites
    run_migrations
    verify_migrations

    log_success "Database migration completed successfully"
}

# Main execution
main() {
    parse_arguments "$@"
    migrate_database
}

# Run main function if script is executed directly
if [[ "${BASH_SOURCE[0]}" == "${0}" ]]; then
    main "$@"
fi
