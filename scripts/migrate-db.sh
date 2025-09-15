#!/bin/bash

# Database Migration Script for Mira Backend
# This script runs database migrations after RDS deployment

set -e  # Exit on any error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

print_status() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

print_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Default values
STAGE="dev"
REGION="us-east-1"
PROFILE="default"

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        -s|--stage)
            STAGE="$2"
            shift 2
            ;;
        -r|--region)
            REGION="$2"
            shift 2
            ;;
        -p|--profile)
            PROFILE="$2"
            shift 2
            ;;
        -h|--help)
            echo "Usage: $0 [OPTIONS]"
            echo "Options:"
            echo "  -s, --stage STAGE     Deployment stage (dev, staging, prod) [default: dev]"
            echo "  -r, --region REGION   AWS region [default: us-east-1]"
            echo "  -p, --profile PROFILE AWS profile [default: default]"
            echo "  -h, --help           Show this help message"
            exit 0
            ;;
        *)
            print_error "Unknown option: $1"
            exit 1
            ;;
    esac
done

print_status "Running database migrations for Mira Backend"
print_status "Stage: $STAGE"
print_status "Region: $REGION"
print_status "Profile: $PROFILE"

# Get RDS endpoint from CloudFormation stack
get_rds_endpoint() {
    print_status "Getting RDS endpoint from CloudFormation stack..."

    local stack_name="mira-api-${STAGE}"
    local db_endpoint

    db_endpoint=$(aws cloudformation describe-stacks \
        --stack-name "$stack_name" \
        --profile "$PROFILE" \
        --region "$REGION" \
        --query "Stacks[0].Outputs[?OutputKey=='DatabaseEndpoint'].OutputValue" \
        --output text 2>/dev/null || echo "")

    if [ -z "$db_endpoint" ] || [ "$db_endpoint" = "None" ]; then
        print_error "Could not retrieve RDS endpoint. Make sure the stack is deployed and accessible."
        print_status "Stack name: $stack_name"
        print_status "Region: $REGION"
        print_status "Profile: $PROFILE"
        exit 1
    fi

    echo "$db_endpoint"
}

# Get database password from environment or use default
get_db_password() {
    local password="${DB_PASSWORD:-MiraDB123!}"
    echo "$password"
}

# Wait for RDS to be available
wait_for_rds() {
    local endpoint="$1"
    local port="5432"

    print_status "Waiting for RDS instance to be available..."

    local max_attempts=30
    local attempt=1

    while [ $attempt -le $max_attempts ]; do
        if nc -z "$endpoint" "$port" 2>/dev/null; then
            print_success "RDS instance is available"
            return 0
        fi

        print_status "Attempt $attempt/$max_attempts: RDS not ready yet, waiting 30 seconds..."
        sleep 30
        ((attempt++))
    done

    print_error "RDS instance did not become available within the timeout period"
    exit 1
}

# Run database migrations
run_migrations() {
    local endpoint="$1"
    local password="$2"

    print_status "Running database migrations..."

    # Set DATABASE_URL for alembic
    export DATABASE_URL="postgresql://mira_user:${password}@${endpoint}:5432/mira_db"

    # Check if alembic is available
    if ! command -v alembic &> /dev/null; then
        print_error "Alembic is not installed. Please install it first:"
        print_status "pip install alembic"
        exit 1
    fi

    # Run migrations
    print_status "Executing: alembic upgrade head"
    alembic upgrade head

    print_success "Database migrations completed successfully"
}

# Test database connection
test_connection() {
    local endpoint="$1"
    local password="$2"

    print_status "Testing database connection..."

    # Test with psql if available
    if command -v psql &> /dev/null; then
        export PGPASSWORD="$password"
        if psql -h "$endpoint" -U mira_user -d mira_db -c "SELECT 1;" &>/dev/null; then
            print_success "Database connection test successful"
        else
            print_warning "Database connection test failed, but migrations may have succeeded"
        fi
    else
        print_warning "psql not available, skipping connection test"
    fi
}

# Main execution
main() {
    local rds_endpoint
    local db_password

    # Get RDS endpoint
    rds_endpoint=$(get_rds_endpoint)
    print_status "RDS Endpoint: $rds_endpoint"

    # Get database password
    db_password=$(get_db_password)

    # Wait for RDS to be available
    wait_for_rds "$rds_endpoint"

    # Run migrations
    run_migrations "$rds_endpoint" "$db_password"

    # Test connection
    test_connection "$rds_endpoint" "$db_password"

    print_success "Database setup completed successfully!"
    print_status "RDS Endpoint: $rds_endpoint"
    print_status "Database: mira_db"
    print_status "Username: mira_user"
    print_status "Port: 5432"
}

# Run main function
main
