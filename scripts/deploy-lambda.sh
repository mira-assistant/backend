#!/bin/bash

# Enterprise-grade Lambda deployment script
# Supports multiple environments and proper error handling

set -euo pipefail  # Exit on error, undefined vars, pipe failures

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Default values
STAGE="${STAGE:-dev}"
REGION="${REGION:-us-east-1}"
FUNCTION_NAME="mira-backend-$STAGE"
MEMORY_SIZE="${MEMORY_SIZE:-1024}"
TIMEOUT="${TIMEOUT:-30}"

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

    # Check if .env file exists
    if [ ! -f ".env" ]; then
        print_error ".env file not found. Please create one with your database URL and API keys."
        exit 1
    fi

    # Load environment variables safely
    set -a
    source .env
    set +a

    # Validate required variables
    if [ -z "${DATABASE_URL:-}" ]; then
        print_error "DATABASE_URL not found in .env file"
        exit 1
    fi

    if [ -z "${GEMINI_API_KEY:-}" ]; then
        print_error "GEMINI_API_KEY not found in .env file"
        exit 1
    fi

    # Validate AWS credentials
    if ! aws sts get-caller-identity >/dev/null 2>&1; then
        print_error "AWS credentials not configured. Please run 'aws configure'"
        exit 1
    fi

    print_success "Environment validation passed"
}

# Main execution
main() {
    print_status "Starting Lambda deployment for $FUNCTION_NAME"

    # Validate environment first
    validate_environment

    # Create deployment package
    print_status "Creating deployment package..."
    mkdir -p deployment
    cp -r app deployment/
    cp lambda_handler.py deployment/
    cp requirements.txt deployment/

    # Install dependencies
    print_status "Installing dependencies..."
    cd deployment
    pip install -r requirements.txt -t . --quiet
    cd ..

    # Create zip file
    print_status "Creating deployment zip..."
    cd deployment
    zip -r ../lambda-deployment.zip . -q
    cd ..
    rm -rf deployment

    # Check if Lambda function exists
    print_status "Checking if Lambda function exists..."
    if aws lambda get-function --function-name $FUNCTION_NAME --region $REGION >/dev/null 2>&1; then
        print_status "Updating existing Lambda function..."
        aws lambda update-function-code \
            --function-name $FUNCTION_NAME \
            --zip-file fileb://lambda-deployment.zip \
            --region $REGION
    else
        print_status "Creating new Lambda function..."
        aws lambda create-function \
            --function-name $FUNCTION_NAME \
            --runtime python3.12 \
            --role arn:aws:iam::$(aws sts get-caller-identity --query Account --output text):role/lambda-execution-role \
            --handler lambda_handler.lambda_handler \
            --zip-file fileb://lambda-deployment.zip \
            --timeout $TIMEOUT \
            --memory-size $MEMORY_SIZE \
            --environment Variables="{
                \"DATABASE_URL\":\"$DATABASE_URL\",
                \"GEMINI_API_KEY\":\"$GEMINI_API_KEY\"
            }" \
            --region $REGION
    fi

    # Clean up
    rm lambda-deployment.zip

    print_success "Lambda function deployed successfully!"
    print_status "Function name: $FUNCTION_NAME"
    print_status "Region: $REGION"
    print_status "You can test it using: aws lambda invoke --function-name $FUNCTION_NAME response.json"
}

# Run main function
main "$@"
