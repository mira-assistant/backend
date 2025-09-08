#!/bin/bash

# Mira Backend Lambda Setup Script
# This script sets up the environment for AWS Lambda deployment

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

print_status "Setting up Mira Backend for AWS Lambda deployment..."

# Check if Node.js is installed
if ! command -v node &> /dev/null; then
    print_error "Node.js is not installed. Please install Node.js first."
    print_status "Visit: https://nodejs.org/"
    exit 1
fi

# Check if Python is installed
if ! command -v python3 &> /dev/null; then
    print_error "Python 3 is not installed. Please install Python 3 first."
    exit 1
fi

# Check if AWS CLI is installed
if ! command -v aws &> /dev/null; then
    print_warning "AWS CLI is not installed. Installing..."
    if [[ "$OSTYPE" == "darwin"* ]]; then
        # macOS
        if command -v brew &> /dev/null; then
            brew install awscli
        else
            print_error "Homebrew not found. Please install AWS CLI manually."
            exit 1
        fi
    elif [[ "$OSTYPE" == "linux-gnu"* ]]; then
        # Linux
        curl "https://awscli.amazonaws.com/awscli-exe-linux-x86_64.zip" -o "awscliv2.zip"
        unzip awscliv2.zip
        sudo ./aws/install
        rm -rf aws awscliv2.zip
    else
        print_error "Unsupported OS. Please install AWS CLI manually."
        exit 1
    fi
fi

# Install Serverless Framework
print_status "Installing Serverless Framework..."
npm install -g serverless

# Install serverless plugins
print_status "Installing Serverless plugins..."
npm install -g serverless-python-requirements
npm install -g serverless-offline

# Install local dependencies
print_status "Installing local dependencies..."
npm install

# Create virtual environment for Python
print_status "Setting up Python virtual environment..."
if [ ! -d "venv" ]; then
    python3 -m venv venv
fi

# Activate virtual environment and install dependencies
print_status "Installing Python dependencies..."
source venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt

# Create .env file if it doesn't exist
if [ ! -f ".env" ]; then
    print_status "Creating .env file from template..."
    if [ -f "env.template" ]; then
        cp env.template .env
        print_warning "Please edit .env file with your actual values before deploying."
    else
        print_error "env.template file not found."
        exit 1
    fi
fi

# Check AWS credentials
print_status "Checking AWS credentials..."
if ! aws sts get-caller-identity &> /dev/null; then
    print_warning "AWS credentials not configured."
    print_status "Please run: aws configure"
    print_status "You'll need:"
    print_status "  - AWS Access Key ID"
    print_status "  - AWS Secret Access Key"
    print_status "  - Default region (e.g., us-east-1)"
    print_status "  - Default output format (json)"
fi

print_success "Setup completed successfully!"
print_status ""
print_status "Next steps:"
print_status "1. Edit .env file with your actual values"
print_status "2. Configure AWS credentials: aws configure"
print_status "3. Deploy to AWS: ./scripts/deploy.sh"
print_status ""
print_status "For detailed instructions, see LAMBDA_DEPLOYMENT_GUIDE.md"
