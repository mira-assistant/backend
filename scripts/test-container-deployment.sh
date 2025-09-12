#!/bin/bash

# Test script for container-based Lambda deployment
# This script tests the deployment locally before pushing to GitHub

set -e

# Configuration
ENVIRONMENT="${1:-dev}"
AWS_REGION="${AWS_REGION:-us-east-1}"

echo "🧪 Testing Container Lambda Deployment"
echo "Environment: ${ENVIRONMENT}"
echo "Region: ${AWS_REGION}"
echo ""

# Check if Docker is running
if ! docker info >/dev/null 2>&1; then
    echo "❌ Docker is not running. Please start Docker and try again."
    exit 1
fi

# Check if AWS CLI is configured
if ! aws sts get-caller-identity >/dev/null 2>&1; then
    echo "❌ AWS CLI is not configured. Please run 'aws configure' first."
    exit 1
fi

# Check if .env file exists
if [ ! -f ".env" ]; then
    echo "❌ .env file not found. Please create one from .env.example"
    exit 1
fi

echo "✅ Prerequisites check passed"
echo ""

# Test Docker build
echo "🐳 Testing Docker build..."
if docker build -f Dockerfile.lambda -t mira-backend-test .; then
    echo "✅ Docker build successful"
else
    echo "❌ Docker build failed"
    exit 1
fi

echo ""
echo "🎉 All tests passed! Ready for deployment."
echo ""
echo "To deploy:"
echo "1. Add GitHub secrets (AWS_ACCESS_KEY_ID, AWS_SECRET_ACCESS_KEY, DATABASE_URL, GEMINI_API_KEY)"
echo "2. Push to GitHub to trigger deployment"
echo "3. Or run locally: ./scripts/deployment/deploy-application-container.sh ${ENVIRONMENT}"
