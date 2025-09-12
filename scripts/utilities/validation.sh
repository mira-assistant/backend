#!/bin/bash

# Enterprise Validation Utilities
# Provides validation functionality for environment, dependencies, and configurations

# Validation functions
validate_required_env_vars() {
    local missing_vars=()
    local var

    for var in "$@"; do
        if [ -z "${!var:-}" ]; then
            missing_vars+=("${var}")
        fi
    done

    if [ ${#missing_vars[@]} -gt 0 ]; then
        log_error "Missing required environment variables:"
        for var in "${missing_vars[@]}"; do
            log_error "  - ${var}"
        done
        return 1
    fi

    return 0
}

validate_aws_credentials() {
    log_info "Validating AWS credentials..."

    if ! command -v aws &> /dev/null; then
        log_error "AWS CLI is not installed"
        return 1
    fi

    if ! aws sts get-caller-identity &> /dev/null; then
        log_error "AWS credentials not configured or invalid"
        log_error "Please run: aws configure"
        return 1
    fi

    local account_id
    account_id=$(aws sts get-caller-identity --query Account --output text)
    log_info "AWS Account ID: ${account_id}"

    return 0
}

validate_aws_region() {
    local region="${1:-${AWS_REGION:-us-east-1}}"

    log_info "Validating AWS region: ${region}"

    if ! aws ec2 describe-regions --region-names "${region}" &> /dev/null; then
        log_error "Invalid AWS region: ${region}"
        return 1
    fi

    log_success "AWS region validation passed"
    return 0
}

validate_environment() {
    local environment="${1:-dev}"

    log_info "Validating environment: ${environment}"

    case "${environment}" in
        "dev"|"development")
            log_info "Development environment detected"
            ;;
        "staging"|"stage")
            log_info "Staging environment detected"
            ;;
        "prod"|"production")
            log_info "Production environment detected"
            ;;
        *)
            log_error "Invalid environment: ${environment}"
            log_error "Valid environments: dev, staging, prod"
            return 1
            ;;
    esac

    return 0
}

validate_file_exists() {
    local file_path="$1"
    local description="${2:-File}"

    if [ ! -f "${file_path}" ]; then
        log_error "${description} not found: ${file_path}"
        return 1
    fi

    log_debug "${description} found: ${file_path}"
    return 0
}

validate_directory_exists() {
    local dir_path="$1"
    local description="${2:-Directory}"

    if [ ! -d "${dir_path}" ]; then
        log_error "${description} not found: ${dir_path}"
        return 1
    fi

    log_debug "${description} found: ${dir_path}"
    return 0
}

validate_python_environment() {
    log_info "Validating Python environment..."

    if ! command -v python3 &> /dev/null; then
        log_error "Python 3 is not installed"
        return 1
    fi

    local python_version
    python_version=$(python3 --version 2>&1 | cut -d' ' -f2)
    log_info "Python version: ${python_version}"

    # Check if virtual environment is active
    if [ -n "${VIRTUAL_ENV:-}" ]; then
        log_info "Virtual environment active: ${VIRTUAL_ENV}"
    else
        log_warning "No virtual environment detected"
    fi

    return 0
}

validate_required_packages() {
    local packages=("$@")
    local missing_packages=()
    local package

    log_info "Validating required Python packages..."

    for package in "${packages[@]}"; do
        if ! python3 -c "import ${package}" &> /dev/null; then
            missing_packages+=("${package}")
        fi
    done

    if [ ${#missing_packages[@]} -gt 0 ]; then
        log_error "Missing required Python packages:"
        for package in "${missing_packages[@]}"; do
            log_error "  - ${package}"
        done
        return 1
    fi

    log_success "All required packages are available"
    return 0
}

validate_database_url() {
    local database_url="${1:-${DATABASE_URL:-}}"

    if [ -z "${database_url}" ]; then
        log_error "DATABASE_URL not provided"
        return 1
    fi

    # Check URL format
    if [[ ! "${database_url}" =~ ^postgresql:// ]]; then
        log_error "Invalid database URL format. Expected: postgresql://..."
        return 1
    fi

    log_debug "Database URL format validation passed"
    return 0
}

validate_lambda_function_name() {
    local function_name="$1"

    # Lambda function name validation rules
    # - Must be 1-64 characters
    # - Can contain letters, numbers, hyphens, and underscores
    # - Cannot start or end with hyphen

    if [ ${#function_name} -lt 1 ] || [ ${#function_name} -gt 64 ]; then
        log_error "Lambda function name must be 1-64 characters: ${function_name}"
        return 1
    fi

    if [[ ! "${function_name}" =~ ^[a-zA-Z0-9][a-zA-Z0-9_-]*[a-zA-Z0-9]$ ]] && [[ ! "${function_name}" =~ ^[a-zA-Z0-9]$ ]]; then
        log_error "Invalid Lambda function name format: ${function_name}"
        log_error "Must contain only letters, numbers, hyphens, and underscores"
        log_error "Cannot start or end with hyphen"
        return 1
    fi

    log_debug "Lambda function name validation passed: ${function_name}"
    return 0
}

validate_memory_size() {
    local memory_size="$1"

    # Lambda memory size validation
    # Must be between 128 and 10240 MB, in 1 MB increments

    if ! [[ "${memory_size}" =~ ^[0-9]+$ ]]; then
        log_error "Memory size must be a number: ${memory_size}"
        return 1
    fi

    if [ "${memory_size}" -lt 128 ] || [ "${memory_size}" -gt 10240 ]; then
        log_error "Memory size must be between 128 and 10240 MB: ${memory_size}"
        return 1
    fi

    if [ $((memory_size % 1)) -ne 0 ]; then
        log_error "Memory size must be in 1 MB increments: ${memory_size}"
        return 1
    fi

    log_debug "Memory size validation passed: ${memory_size} MB"
    return 0
}

validate_timeout() {
    local timeout="$1"

    # Lambda timeout validation
    # Must be between 1 and 900 seconds

    if ! [[ "${timeout}" =~ ^[0-9]+$ ]]; then
        log_error "Timeout must be a number: ${timeout}"
        return 1
    fi

    if [ "${timeout}" -lt 1 ] || [ "${timeout}" -gt 900 ]; then
        log_error "Timeout must be between 1 and 900 seconds: ${timeout}"
        return 1
    fi

    log_debug "Timeout validation passed: ${timeout} seconds"
    return 0
}

validate_github_secrets() {
    local required_secrets=("$@")
    local missing_secrets=()
    local secret

    log_info "Validating GitHub secrets..."

    for secret in "${required_secrets[@]}"; do
        if [ -z "${!secret:-}" ]; then
            missing_secrets+=("${secret}")
        fi
    done

    if [ ${#missing_secrets[@]} -gt 0 ]; then
        log_error "Missing required GitHub secrets:"
        for secret in "${missing_secrets[@]}"; do
            log_error "  - ${secret}"
        done
        log_error "Please configure these secrets in your GitHub repository settings"
        return 1
    fi

    log_success "All required GitHub secrets are configured"
    return 0
}

validate_deployment_prerequisites() {
    local environment="$1"

    log_info "Validating deployment prerequisites for environment: ${environment}"

    # Validate environment
    validate_environment "${environment}" || return 1

    # Validate AWS credentials
    validate_aws_credentials || return 1

    # Validate AWS region
    validate_aws_region || return 1

    # Validate required environment variables
    validate_required_env_vars "DATABASE_URL" "GEMINI_API_KEY" || return 1

    # Validate database URL format
    validate_database_url || return 1

    # Validate Python environment
    validate_python_environment || return 1

    # Validate required packages
    validate_required_packages "psycopg2" "alembic" || return 1

    log_success "All deployment prerequisites validated"
    return 0
}

# Comprehensive validation for CI/CD pipeline
validate_ci_prerequisites() {
    log_info "Validating CI prerequisites..."

    # Validate Python environment
    validate_python_environment || return 1

    # Validate required packages for testing
    validate_required_packages "pytest" "flake8" "black" "isort" "mypy" || return 1

    # Validate project structure
    validate_file_exists "requirements.txt" "Requirements file" || return 1
    validate_file_exists "pytest.ini" "Pytest configuration" || return 1
    validate_directory_exists "app" "Application directory" || return 1
    validate_directory_exists "tests" "Tests directory" || return 1

    log_success "All CI prerequisites validated"
    return 0
}
