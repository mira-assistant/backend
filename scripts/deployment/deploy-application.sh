#!/bin/bash

# Enterprise Application Deployment Script
# Deploys Mira Backend application to AWS Lambda

set -euo pipefail

# Script configuration
readonly SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
readonly PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
readonly SCRIPT_NAME="$(basename "$0")"

# Load utilities
source "${SCRIPT_DIR}/../utilities/logging.sh"
source "${SCRIPT_DIR}/../utilities/validation.sh"
source "${SCRIPT_DIR}/../utilities/aws-utils.sh"

# Default configuration
readonly DEFAULT_ENVIRONMENT="dev"
readonly DEFAULT_REGION="us-east-1"
readonly DEFAULT_MEMORY_SIZE="1024"
readonly DEFAULT_TIMEOUT="30"

# Parse command line arguments
parse_arguments() {
    local environment="${1:-${DEFAULT_ENVIRONMENT}}"
    local region="${2:-${DEFAULT_REGION}}"
    local memory_size="${3:-${DEFAULT_MEMORY_SIZE}}"
    local timeout="${4:-${DEFAULT_TIMEOUT}}"

    log_info "Starting application deployment"
    log_info "Environment: ${environment}"
    log_info "Region: ${region}"
    log_info "Memory Size: ${memory_size}MB"
    log_info "Timeout: ${timeout}s"

    # Export for use in other functions
    export ENVIRONMENT="${environment}"
    export AWS_REGION="${region}"
    export MEMORY_SIZE="${memory_size}"
    export TIMEOUT="${timeout}"
    export FUNCTION_NAME="mira-backend-${environment}"
}

# Validate prerequisites
validate_prerequisites() {
    log_info "Validating prerequisites..."

    # Check AWS CLI
    if ! command -v aws &> /dev/null; then
        log_error "AWS CLI is not installed"
        exit 1
    fi

    # Check AWS credentials
    if ! aws sts get-caller-identity &> /dev/null; then
        log_error "AWS credentials not configured"
        exit 1
    fi

    # Check required environment variables
    validate_required_env_vars "DATABASE_URL" "GEMINI_API_KEY"

    # Check if Lambda role exists
    local role_name="mira-lambda-execution-role-${ENVIRONMENT}"
    if ! aws iam get-role --role-name "${role_name}" &> /dev/null; then
        log_error "Lambda execution role not found: ${role_name}"
        log_error "Please run infrastructure deployment first"
        exit 1
    fi

    log_success "Prerequisites validated"
}

# Create deployment package
create_deployment_package() {
    log_info "Creating deployment package..."

    local package_dir="${PROJECT_ROOT}/.deployment"
    local package_file="${PROJECT_ROOT}/lambda-deployment-${ENVIRONMENT}.zip"

    # Clean up previous deployment
    rm -rf "${package_dir}"
    rm -f "${package_file}"

    # Create deployment directory
    mkdir -p "${package_dir}"

    # Copy application code
    cp -r "${PROJECT_ROOT}/app" "${package_dir}/"
    cp "${PROJECT_ROOT}/lambda_handler.py" "${package_dir}/"
    cp "${PROJECT_ROOT}/requirements.txt" "${package_dir}/"

    # Install dependencies
    log_info "Installing Python dependencies..."
    cd "${package_dir}"
    pip install -r requirements.txt -t . --quiet --no-cache-dir

    # Create zip package
    log_info "Creating deployment package..."
    zip -r "${package_file}" . -q

    # Clean up
    cd "${PROJECT_ROOT}"
    rm -rf "${package_dir}"

    # Verify package size
    local package_size=$(du -h "${package_file}" | cut -f1)
    log_info "Deployment package created: ${package_file} (${package_size})"

    # Export package file path
    export PACKAGE_FILE="${package_file}"
}

# Deploy or update Lambda function
deploy_lambda_function() {
    log_info "Deploying Lambda function: ${FUNCTION_NAME}"

    local role_arn="arn:aws:iam::$(aws sts get-caller-identity --query Account --output text):role/mira-lambda-execution-role-${ENVIRONMENT}"

    # Check if function exists
    if aws lambda get-function --function-name "${FUNCTION_NAME}" --region "${AWS_REGION}" &> /dev/null; then
        log_info "Updating existing Lambda function..."
        aws lambda update-function-code \
            --function-name "${FUNCTION_NAME}" \
            --zip-file "fileb://${PACKAGE_FILE}" \
            --region "${AWS_REGION}"

        # Update function configuration
        aws lambda update-function-configuration \
            --function-name "${FUNCTION_NAME}" \
            --memory-size "${MEMORY_SIZE}" \
            --timeout "${TIMEOUT}" \
            --environment Variables="{
                \"DATABASE_URL\":\"${DATABASE_URL}\",
                \"GEMINI_API_KEY\":\"${GEMINI_API_KEY}\",
                \"ENVIRONMENT\":\"${ENVIRONMENT}\"
            }" \
            --region "${AWS_REGION}"

        log_success "Lambda function updated successfully"
    else
        log_info "Creating new Lambda function..."
        aws lambda create-function \
            --function-name "${FUNCTION_NAME}" \
            --runtime python3.12 \
            --role "${role_arn}" \
            --handler lambda_handler.lambda_handler \
            --zip-file "fileb://${PACKAGE_FILE}" \
            --memory-size "${MEMORY_SIZE}" \
            --timeout "${TIMEOUT}" \
            --environment Variables="{
                \"DATABASE_URL\":\"${DATABASE_URL}\",
                \"GEMINI_API_KEY\":\"${GEMINI_API_KEY}\",
                \"ENVIRONMENT\":\"${ENVIRONMENT}\"
            }" \
            --region "${AWS_REGION}"

        log_success "Lambda function created successfully"
    fi
}

# Create or update API Gateway
deploy_api_gateway() {
    log_info "Deploying API Gateway..."

    local api_name="mira-backend-api-${ENVIRONMENT}"
    local api_id

    # Check if API exists
    api_id=$(aws apigatewayv2 get-apis --query "Items[?Name=='${api_name}'].ApiId" --output text --region "${AWS_REGION}")

    if [ -z "${api_id}" ] || [ "${api_id}" = "None" ]; then
        log_info "Creating new API Gateway..."
        api_id=$(aws apigatewayv2 create-api \
            --name "${api_name}" \
            --protocol-type HTTP \
            --region "${AWS_REGION}" \
            --query 'ApiId' --output text)

        log_success "API Gateway created: ${api_id}"
    else
        log_info "API Gateway already exists: ${api_id}"
    fi

    # Export API ID for health checks
    export API_ID="${api_id}"
}

# Clean up deployment artifacts
cleanup() {
    log_info "Cleaning up deployment artifacts..."

    if [ -n "${PACKAGE_FILE:-}" ] && [ -f "${PACKAGE_FILE}" ]; then
        rm -f "${PACKAGE_FILE}"
        log_info "Deployment package cleaned up"
    fi
}

# Main deployment function
deploy_application() {
    log_info "Deploying application for environment: ${ENVIRONMENT}"

    validate_prerequisites
    create_deployment_package
    deploy_lambda_function
    deploy_api_gateway
    cleanup

    log_success "Application deployment completed successfully"
    log_info "Function name: ${FUNCTION_NAME}"
    log_info "Region: ${AWS_REGION}"
    log_info "API Gateway ID: ${API_ID:-N/A}"
}

# Main execution
main() {
    parse_arguments "$@"
    deploy_application
}

# Run main function if script is executed directly
if [[ "${BASH_SOURCE[0]}" == "${0}" ]]; then
    main "$@"
fi
