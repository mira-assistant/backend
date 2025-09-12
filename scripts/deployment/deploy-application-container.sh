#!/bin/bash

# Container-based Lambda deployment script for Mira backend
# This script builds a Docker image and deploys it to AWS Lambda

set -e

# Source utility functions
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
UTILITIES_DIR="$PROJECT_ROOT/scripts/utilities"

# Source logging and validation utilities
source "$UTILITIES_DIR/logging.sh"
source "$UTILITIES_DIR/validation.sh"

# Configuration
ENVIRONMENT="${1:-dev}"
AWS_REGION="${AWS_REGION:-us-east-1}"
FUNCTION_NAME="mira-backend-${ENVIRONMENT}"
ECR_REPOSITORY="mira-backend-${ENVIRONMENT}"
IMAGE_TAG="latest"
MEMORY_SIZE="${MEMORY_SIZE:-1024}"
TIMEOUT="${TIMEOUT:-30}"

# Derived values
ACCOUNT_ID=$(aws sts get-caller-identity --query Account --output text)
ECR_URI="${ACCOUNT_ID}.dkr.ecr.${AWS_REGION}.amazonaws.com/${ECR_REPOSITORY}"
FULL_IMAGE_URI="${ECR_URI}:${IMAGE_TAG}"

log_info "Starting container-based application deployment"
log_info "Environment: ${ENVIRONMENT}"
log_info "Region: ${AWS_REGION}"
log_info "Function Name: ${FUNCTION_NAME}"
log_info "ECR Repository: ${ECR_REPOSITORY}"
log_info "Image URI: ${FULL_IMAGE_URI}"

# Validate prerequisites
log_info "Validating prerequisites..."
validate_aws_credentials
validate_required_env_vars

# Create ECR repository if it doesn't exist
log_info "Ensuring ECR repository exists..."
aws ecr describe-repositories --repository-names "${ECR_REPOSITORY}" --region "${AWS_REGION}" >/dev/null 2>&1 || {
    log_info "Creating ECR repository: ${ECR_REPOSITORY}"
    aws ecr create-repository \
        --repository-name "${ECR_REPOSITORY}" \
        --region "${AWS_REGION}" \
        --image-scanning-configuration scanOnPush=true
}

# Get ECR login token
log_info "Logging into ECR..."
aws ecr get-login-password --region "${AWS_REGION}" | docker login --username AWS --password-stdin "${ECR_URI}"

# Build Docker image
log_info "Building Docker image..."
cd "${PROJECT_ROOT}"
docker build -f Dockerfile.lambda -t "${ECR_REPOSITORY}:${IMAGE_TAG}" .

# Tag image for ECR
docker tag "${ECR_REPOSITORY}:${IMAGE_TAG}" "${FULL_IMAGE_URI}"

# Push image to ECR
log_info "Pushing image to ECR..."
docker push "${FULL_IMAGE_URI}"

# Get or create Lambda execution role
log_info "Setting up Lambda execution role..."
ROLE_NAME="mira-lambda-execution-role-${ENVIRONMENT}"
ROLE_ARN="arn:aws:iam::${ACCOUNT_ID}:role/${ROLE_NAME}"

# Check if role exists
if ! aws iam get-role --role-name "${ROLE_NAME}" >/dev/null 2>&1; then
    log_info "Creating Lambda execution role..."

    # Create trust policy
    cat > /tmp/trust-policy.json << EOF
{
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
}
EOF

    # Create role
    aws iam create-role \
        --role-name "${ROLE_NAME}" \
        --assume-role-policy-document file:///tmp/trust-policy.json

    # Attach basic execution policy
    aws iam attach-role-policy \
        --role-name "${ROLE_NAME}" \
        --policy-arn arn:aws:iam::aws:policy/service-role/AWSLambdaBasicExecutionRole

    # Attach additional policies for RDS and ECR
    aws iam attach-role-policy \
        --role-name "${ROLE_NAME}" \
        --policy-arn arn:aws:iam::aws:policy/AmazonRDSFullAccess

    aws iam attach-role-policy \
        --role-name "${ROLE_NAME}" \
        --policy-arn arn:aws:iam::aws:policy/AmazonEC2ContainerRegistryReadOnly

    # Wait for role to be ready
    log_info "Waiting for role to be ready..."
    aws iam wait role-exists --role-name "${ROLE_NAME}"
fi

# Create or update Lambda function
log_info "Deploying Lambda function..."

if aws lambda get-function --function-name "${FUNCTION_NAME}" --region "${AWS_REGION}" >/dev/null 2>&1; then
    log_info "Updating existing Lambda function..."
    aws lambda update-function-code \
        --function-name "${FUNCTION_NAME}" \
        --image-uri "${FULL_IMAGE_URI}" \
        --region "${AWS_REGION}"

    aws lambda update-function-configuration \
        --function-name "${FUNCTION_NAME}" \
        --memory-size "${MEMORY_SIZE}" \
        --timeout "${TIMEOUT}" \
        --environment "Variables={DATABASE_URL=${DATABASE_URL},GEMINI_API_KEY=${GEMINI_API_KEY},ENVIRONMENT=${ENVIRONMENT}}" \
        --region "${AWS_REGION}"
else
    log_info "Creating new Lambda function..."
    aws lambda create-function \
        --function-name "${FUNCTION_NAME}" \
        --package-type Image \
        --code ImageUri="${FULL_IMAGE_URI}" \
        --role "${ROLE_ARN}" \
        --memory-size "${MEMORY_SIZE}" \
        --timeout "${TIMEOUT}" \
        --environment "Variables={DATABASE_URL=${DATABASE_URL},GEMINI_API_KEY=${GEMINI_API_KEY},ENVIRONMENT=${ENVIRONMENT}}" \
        --region "${AWS_REGION}"
fi

# Create API Gateway (if needed)
log_info "Setting up API Gateway..."
REST_API_NAME="mira-api-${ENVIRONMENT}"

# Check if API exists
API_ID=$(aws apigateway get-rest-apis --query "items[?name=='${REST_API_NAME}'].id" --output text --region "${AWS_REGION}")

if [ -z "${API_ID}" ] || [ "${API_ID}" = "None" ]; then
    log_info "Creating API Gateway..."
    API_ID=$(aws apigateway create-rest-api \
        --name "${REST_API_NAME}" \
        --description "Mira Backend API for ${ENVIRONMENT}" \
        --region "${AWS_REGION}" \
        --query 'id' --output text)
fi

# Get root resource ID
ROOT_RESOURCE_ID=$(aws apigateway get-resources \
    --rest-api-id "${API_ID}" \
    --region "${AWS_REGION}" \
    --query 'items[?path==`/`].id' --output text)

# Create proxy resource
PROXY_RESOURCE_ID=$(aws apigateway create-resource \
    --rest-api-id "${API_ID}" \
    --parent-id "${ROOT_RESOURCE_ID}" \
    --path-part "{proxy+}" \
    --region "${AWS_REGION}" \
    --query 'id' --output text)

# Create ANY method
aws apigateway put-method \
    --rest-api-id "${API_ID}" \
    --resource-id "${PROXY_RESOURCE_ID}" \
    --http-method ANY \
    --authorization-type NONE \
    --region "${AWS_REGION}"

# Set up Lambda integration
aws apigateway put-integration \
    --rest-api-id "${API_ID}" \
    --resource-id "${PROXY_RESOURCE_ID}" \
    --http-method ANY \
    --type AWS_PROXY \
    --integration-http-method POST \
    --uri "arn:aws:apigateway:${AWS_REGION}:lambda:path/2015-03-31/functions/arn:aws:lambda:${AWS_REGION}:${ACCOUNT_ID}:function:${FUNCTION_NAME}/invocations" \
    --region "${AWS_REGION}"

# Add Lambda permission for API Gateway
aws lambda add-permission \
    --function-name "${FUNCTION_NAME}" \
    --statement-id "api-gateway-invoke-${ENVIRONMENT}" \
    --action lambda:InvokeFunction \
    --principal apigateway.amazonaws.com \
    --source-arn "arn:aws:execute-api:${AWS_REGION}:${ACCOUNT_ID}:${API_ID}/*/*" \
    --region "${AWS_REGION}" 2>/dev/null || true

# Deploy API
aws apigateway create-deployment \
    --rest-api-id "${API_ID}" \
    --stage-name "${ENVIRONMENT}" \
    --region "${AWS_REGION}"

# Get API URL
API_URL="https://${API_ID}.execute-api.${AWS_REGION}.amazonaws.com/${ENVIRONMENT}"

log_success "Deployment completed successfully!"
log_success "Lambda Function: ${FUNCTION_NAME}"
log_success "API Gateway URL: ${API_URL}"
log_success "ECR Image: ${FULL_IMAGE_URI}"

# Clean up temporary files
rm -f /tmp/trust-policy.json

echo ""
echo "🎉 Container-based Lambda deployment completed!"
echo "📡 API URL: ${API_URL}"
echo "🔧 Function: ${FUNCTION_NAME}"
echo "🐳 Image: ${FULL_IMAGE_URI}"
