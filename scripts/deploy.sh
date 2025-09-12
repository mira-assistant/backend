#!/bin/bash

# Mira Backend AWS Lambda Deployment Script
# This script automates the deployment process to AWS Lambda

set -e  # Exit on any error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Default values
STAGE="dev"
REGION="us-east-1"
PROFILE="default"

# Function to print colored output
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

# Function to show usage
show_usage() {
    echo "Usage: $0 [OPTIONS]"
    echo ""
    echo "Options:"
    echo "  -s, --stage STAGE     Deployment stage (dev, staging, prod) [default: dev]"
    echo "  -r, --region REGION   AWS region [default: us-east-1]"
    echo "  -p, --profile PROFILE AWS profile [default: default]"
    echo "  -h, --help           Show this help message"
    echo ""
    echo "Examples:"
    echo "  $0                                    # Deploy to dev stage"
    echo "  $0 -s prod -r us-west-2              # Deploy to prod stage in us-west-2"
    echo "  $0 -s staging -p mira-profile         # Deploy to staging with custom profile"
}

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
            show_usage
            exit 0
            ;;
        *)
            print_error "Unknown option: $1"
            show_usage
            exit 1
            ;;
    esac
done

print_status "Starting Mira Backend deployment to AWS Lambda"
print_status "Stage: $STAGE"
print_status "Region: $REGION"
print_status "Profile: $PROFILE"

# Check if required tools are installed
check_dependencies() {
    print_status "Checking dependencies..."

    if ! command -v aws &> /dev/null; then
        print_error "AWS CLI is not installed. Please install it first."
        exit 1
    fi

    if ! command -v serverless &> /dev/null; then
        print_error "Serverless Framework is not installed. Installing..."
        npm install -g serverless
    fi

    if ! command -v python3 &> /dev/null; then
        print_error "Python 3 is not installed. Please install it first."
        exit 1
    fi

    print_success "All dependencies are available"
}

# Check AWS credentials
check_aws_credentials() {
    print_status "Checking AWS credentials..."

    if ! aws sts get-caller-identity --profile $PROFILE &> /dev/null; then
        print_error "AWS credentials not configured for profile: $PROFILE"
        print_status "Please run: aws configure --profile $PROFILE"
        exit 1
    fi

    print_success "AWS credentials are valid"
}

# Check if .env file exists
check_env_file() {
    print_status "Checking environment configuration..."

    if [ ! -f ".env" ]; then
        print_warning ".env file not found. Creating from template..."
        if [ -f "env.template" ]; then
            cp env.template .env
            print_warning "Please edit .env file with your actual values before continuing."
            print_warning "Press Enter when you're ready to continue, or Ctrl+C to exit."
            read
        else
            print_error "env.template file not found. Please create a .env file manually."
            exit 1
        fi
    fi

    print_success "Environment configuration found"
}

# Install Python dependencies
install_dependencies() {
    print_status "Installing Python dependencies..."

    if [ ! -d ".venv" ]; then
        print_status "Creating virtual environment..."
        python3 -m venv .venv
    fi

    source .venv/bin/activate
    pip install --upgrade pip
    pip install -r requirements.txt

    print_success "Dependencies installed"
}

# Deploy to AWS Lambda
deploy_lambda() {
    print_status "Deploying to AWS Lambda..."

    # Set environment variables for serverless
    export STAGE=$STAGE
    export REGION=$REGION

    # Deploy using serverless framework
    serverless deploy --stage $STAGE --region $REGION --aws-profile $PROFILE

    print_success "Deployment completed successfully!"
}

# Run database migrations
run_migrations() {
    print_status "Running database migrations..."

    # Wait a bit for RDS to be fully available
    print_status "Waiting for RDS to be ready..."
    sleep 60

    # Run migrations
    ./scripts/migrate-db.sh --stage $STAGE --region $REGION --profile $PROFILE

    print_success "Database migrations completed"
}

# Test the deployment
test_deployment() {
    print_status "Testing deployment..."

    # Get the API Gateway URL
    API_URL=$(aws apigatewayv2 get-apis --profile $PROFILE --region $REGION --query "Items[?Name=='mira-backend-$STAGE'].ApiEndpoint" --output text)

    if [ -z "$API_URL" ]; then
        print_warning "Could not retrieve API Gateway URL. Please check the deployment."
        return
    fi

    print_status "API Gateway URL: $API_URL"

    # Test the root endpoint
    if curl -s "$API_URL/" > /dev/null; then
        print_success "API is responding correctly"
        print_status "You can test your API at: $API_URL"
    else
        print_warning "API test failed. Please check the CloudWatch logs."
    fi
}

# Show deployment summary
show_summary() {
    print_success "Deployment Summary:"
    echo "=================="
    echo "Stage: $STAGE"
    echo "Region: $REGION"
    echo "Profile: $PROFILE"
    echo ""
    echo "Next steps:"
    echo "1. Set up your RDS database (if not already done)"
    echo "2. Run database migrations: alembic upgrade head"
    echo "3. Configure your LLM backend in the Lambda environment variables"
    echo "4. Test your API endpoints"
    echo ""
    echo "Useful commands:"
    echo "- View logs: serverless logs -f api --stage $STAGE --region $REGION"
    echo "- Remove deployment: serverless remove --stage $STAGE --region $REGION"
    echo "- Update environment: serverless deploy function -f api --stage $STAGE --region $REGION"
}

# Main execution
main() {
    check_dependencies
    check_aws_credentials
    check_env_file
    install_dependencies
    deploy_lambda
    run_migrations
    test_deployment
    show_summary
}

# Run main function
main
