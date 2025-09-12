#!/bin/bash

# Enterprise Infrastructure Deployment Script
# Manages AWS infrastructure resources for Mira Backend

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
readonly DEFAULT_STACK_NAME_PREFIX="mira-backend"

# Parse command line arguments
parse_arguments() {
    local environment="${1:-${DEFAULT_ENVIRONMENT}}"
    local region="${2:-${DEFAULT_REGION}}"

    log_info "Starting infrastructure deployment"
    log_info "Environment: ${environment}"
    log_info "Region: ${region}"

    # Export for use in other functions
    export ENVIRONMENT="${environment}"
    export AWS_REGION="${region}"
    export STACK_NAME="${DEFAULT_STACK_NAME_PREFIX}-${environment}"
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

    log_success "Prerequisites validated"
}

# Create IAM role for Lambda
create_lambda_execution_role() {
    log_info "Creating Lambda execution role..."

    local role_name="mira-lambda-execution-role-${ENVIRONMENT}"
    local policy_document='{
        "Version": "2012-10-17",
        "Statement": [
            {
                "Effect": "Allow",
                "Principal": {
                    "Service": "lambda.amazonaws.com"
                },
                "Action": "sts:AssumeRole"
            }
        ]
    }'

    # Create role if it doesn't exist
    if ! aws iam get-role --role-name "${role_name}" &> /dev/null; then
        aws iam create-role \
            --role-name "${role_name}" \
            --assume-role-policy-document "${policy_document}" \
            --description "Execution role for Mira Backend Lambda function"

        # Attach basic execution policy
        aws iam attach-role-policy \
            --role-name "${role_name}" \
            --policy-arn "arn:aws:iam::aws:policy/service-role/AWSLambdaBasicExecutionRole"

        # Attach RDS access policy
        aws iam attach-role-policy \
            --role-name "${role_name}" \
            --policy-arn "arn:aws:iam::aws:policy/AmazonRDSFullAccess"

        log_success "Lambda execution role created: ${role_name}"
    else
        log_info "Lambda execution role already exists: ${role_name}"
    fi

    # Export role ARN for use in application deployment
    export LAMBDA_ROLE_ARN="arn:aws:iam::$(aws sts get-caller-identity --query Account --output text):role/${role_name}"
}

# Create CloudWatch Log Group
create_log_group() {
    log_info "Creating CloudWatch log group..."

    local log_group_name="/aws/lambda/mira-backend-${ENVIRONMENT}"

    if ! aws logs describe-log-groups --log-group-name-prefix "${log_group_name}" --query 'logGroups[0].logGroupName' --output text | grep -q "${log_group_name}"; then
        aws logs create-log-group \
            --log-group-name "${log_group_name}" \
            --region "${AWS_REGION}"

        # Set retention policy (30 days)
        aws logs put-retention-policy \
            --log-group-name "${log_group_name}" \
            --retention-in-days 30 \
            --region "${AWS_REGION}"

        log_success "CloudWatch log group created: ${log_group_name}"
    else
        log_info "CloudWatch log group already exists: ${log_group_name}"
    fi
}

# Main deployment function
deploy_infrastructure() {
    log_info "Deploying infrastructure for environment: ${ENVIRONMENT}"

    validate_prerequisites
    create_lambda_execution_role
    create_log_group

    log_success "Infrastructure deployment completed successfully"
}

# Main execution
main() {
    parse_arguments "$@"
    deploy_infrastructure
}

# Run main function if script is executed directly
if [[ "${BASH_SOURCE[0]}" == "${0}" ]]; then
    main "$@"
fi
