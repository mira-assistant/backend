#!/bin/bash

# Exit on any error
set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Function to print colored output
print_status() {
    echo -e "${GREEN}[INFO]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Check if required environment variables are set
check_env() {
    if [ -z "$AWS_REGION" ]; then
        print_error "AWS_REGION environment variable is not set"
        exit 1
    fi

    if [ -z "$AWS_ACCESS_KEY_ID" ]; then
        print_error "AWS_ACCESS_KEY_ID environment variable is not set"
        exit 1
    fi

    if [ -z "$AWS_SECRET_ACCESS_KEY" ]; then
        print_error "AWS_SECRET_ACCESS_KEY environment variable is not set"
        exit 1
    fi
}

# Build and push Docker image
build_and_push() {
    print_status "Building Docker image..."
    docker build -t mira-backend:latest .

    print_status "Tagging image for ECR..."
    docker tag mira-backend:latest $ECR_REGISTRY/$ECR_REPOSITORY:latest

    print_status "Pushing image to ECR..."
    docker push $ECR_REGISTRY/$ECR_REPOSITORY:latest
}

# Deploy to ECS
deploy_to_ecs() {
    print_status "Deploying to ECS..."

    # Update ECS service
    aws ecs update-service \
        --cluster $ECS_CLUSTER \
        --service $ECS_SERVICE \
        --force-new-deployment \
        --region $AWS_REGION

    # Wait for deployment to complete
    print_status "Waiting for deployment to complete..."
    aws ecs wait services-stable \
        --cluster $ECS_CLUSTER \
        --services $ECS_SERVICE \
        --region $AWS_REGION

    print_status "Deployment completed successfully!"
}

# Run database migrations
run_migrations() {
    print_status "Running database migrations..."

    # Get the task ARN
    TASK_ARN=$(aws ecs list-tasks \
        --cluster $ECS_CLUSTER \
        --service-name $ECS_SERVICE \
        --query 'taskArns[0]' \
        --output text \
        --region $AWS_REGION)

    if [ "$TASK_ARN" = "None" ] || [ -z "$TASK_ARN" ]; then
        print_error "No running tasks found for service $ECS_SERVICE"
        exit 1
    fi

    # Run migrations
    aws ecs execute-command \
        --cluster $ECS_CLUSTER \
        --task $TASK_ARN \
        --container mira-backend \
        --interactive \
        --command "alembic upgrade head" \
        --region $AWS_REGION
}

# Main deployment function
main() {
    print_status "Starting deployment process..."

    # Check environment
    check_env

    # Set default values if not provided
    ECR_REGISTRY=${ECR_REGISTRY:-"$AWS_ACCOUNT_ID.dkr.ecr.$AWS_REGION.amazonaws.com"}
    ECR_REPOSITORY=${ECR_REPOSITORY:-"mira-backend"}
    ECS_CLUSTER=${ECS_CLUSTER:-"mira-cluster"}
    ECS_SERVICE=${ECS_SERVICE:-"mira-backend-service"}

    # Login to ECR
    print_status "Logging in to ECR..."
    aws ecr get-login-password --region $AWS_REGION | \
        docker login --username AWS --password-stdin $ECR_REGISTRY

    # Build and push image
    build_and_push

    # Deploy to ECS
    deploy_to_ecs

    # Run migrations
    run_migrations

    print_status "Deployment completed successfully!"
}

# Run main function
main "$@"
