#!/bin/bash

# Enterprise Logging Utilities
# Provides consistent logging functionality across all scripts

# Log levels
readonly LOG_LEVEL_DEBUG=0
readonly LOG_LEVEL_INFO=1
readonly LOG_LEVEL_WARNING=2
readonly LOG_LEVEL_ERROR=3

# Default log level
LOG_LEVEL=${LOG_LEVEL:-${LOG_LEVEL_INFO}}

# Colors for output
readonly RED='\033[0;31m'
readonly GREEN='\033[0;32m'
readonly YELLOW='\033[1;33m'
readonly BLUE='\033[0;34m'
readonly PURPLE='\033[0;35m'
readonly CYAN='\033[0;36m'
readonly NC='\033[0m' # No Color

# Logging functions
log_debug() {
    if [ ${LOG_LEVEL} -le ${LOG_LEVEL_DEBUG} ]; then
        echo -e "${PURPLE}[DEBUG]${NC} $(date '+%Y-%m-%d %H:%M:%S') - $1" >&2
    fi
}

log_info() {
    if [ ${LOG_LEVEL} -le ${LOG_LEVEL_INFO} ]; then
        echo -e "${BLUE}[INFO]${NC} $(date '+%Y-%m-%d %H:%M:%S') - $1"
    fi
}

log_warning() {
    if [ ${LOG_LEVEL} -le ${LOG_LEVEL_WARNING} ]; then
        echo -e "${YELLOW}[WARNING]${NC} $(date '+%Y-%m-%d %H:%M:%S') - $1" >&2
    fi
}

log_error() {
    if [ ${LOG_LEVEL} -le ${LOG_LEVEL_ERROR} ]; then
        echo -e "${RED}[ERROR]${NC} $(date '+%Y-%m-%d %H:%M:%S') - $1" >&2
    fi
}

log_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $(date '+%Y-%m-%d %H:%M:%S') - $1"
}

# Log with context
log_with_context() {
    local level="$1"
    local context="$2"
    local message="$3"

    case "${level}" in
        "debug")
            log_debug "[${context}] ${message}"
            ;;
        "info")
            log_info "[${context}] ${message}"
            ;;
        "warning")
            log_warning "[${context}] ${message}"
            ;;
        "error")
            log_error "[${context}] ${message}"
            ;;
        "success")
            log_success "[${context}] ${message}"
            ;;
        *)
            log_info "[${context}] ${message}"
            ;;
    esac
}

# Log function entry/exit
log_function_entry() {
    local function_name="$1"
    log_debug "Entering function: ${function_name}"
}

log_function_exit() {
    local function_name="$1"
    local exit_code="${2:-0}"
    if [ ${exit_code} -eq 0 ]; then
        log_debug "Exiting function: ${function_name} (success)"
    else
        log_debug "Exiting function: ${function_name} (failed with code: ${exit_code})"
    fi
}

# Log command execution
log_command() {
    local command="$1"
    log_debug "Executing command: ${command}"
}

# Log file operations
log_file_operation() {
    local operation="$1"
    local file_path="$2"
    log_debug "File ${operation}: ${file_path}"
}

# Log environment information
log_environment() {
    log_info "Environment Information:"
    log_info "  - Script: ${SCRIPT_NAME:-unknown}"
    log_info "  - Working Directory: $(pwd)"
    log_info "  - User: $(whoami)"
    log_info "  - Hostname: $(hostname)"
    log_info "  - OS: $(uname -s)"
    log_info "  - Architecture: $(uname -m)"
    log_info "  - Shell: ${SHELL}"
    log_info "  - Log Level: ${LOG_LEVEL}"
}

# Log AWS information
log_aws_info() {
    if command -v aws &> /dev/null; then
        log_info "AWS Information:"
        log_info "  - AWS CLI Version: $(aws --version 2>&1 | head -n1)"
        log_info "  - AWS Region: ${AWS_REGION:-not set}"
        log_info "  - AWS Profile: ${AWS_PROFILE:-default}"

        # Get AWS account ID if possible
        if aws sts get-caller-identity &> /dev/null; then
            local account_id
            account_id=$(aws sts get-caller-identity --query Account --output text 2>/dev/null || echo "unknown")
            log_info "  - AWS Account ID: ${account_id}"
        else
            log_warning "AWS credentials not configured or invalid"
        fi
    else
        log_warning "AWS CLI not found"
    fi
}

# Log deployment summary
log_deployment_summary() {
    local environment="$1"
    local function_name="$2"
    local region="$3"
    local api_id="${4:-N/A}"

    log_success "Deployment Summary:"
    log_info "  - Environment: ${environment}"
    log_info "  - Function Name: ${function_name}"
    log_info "  - Region: ${region}"
    log_info "  - API Gateway ID: ${api_id}"
    log_info "  - Timestamp: $(date -u +"%Y-%m-%dT%H:%M:%SZ")"
}

# Log error with stack trace
log_error_with_stack() {
    local error_message="$1"
    log_error "${error_message}"
    log_error "Stack trace:"
    local i=0
    while caller ${i}; do
        ((i++))
    done >&2
}

# Log performance metrics
log_performance() {
    local operation="$1"
    local start_time="$2"
    local end_time="$3"
    local duration=$((end_time - start_time))

    log_info "Performance: ${operation} took ${duration} seconds"
}

# Set log level from environment or argument
set_log_level() {
    local level="$1"
    case "${level}" in
        "debug"|"DEBUG")
            LOG_LEVEL=${LOG_LEVEL_DEBUG}
            ;;
        "info"|"INFO")
            LOG_LEVEL=${LOG_LEVEL_INFO}
            ;;
        "warning"|"WARNING")
            LOG_LEVEL=${LOG_LEVEL_WARNING}
            ;;
        "error"|"ERROR")
            LOG_LEVEL=${LOG_LEVEL_ERROR}
            ;;
        *)
            log_warning "Unknown log level: ${level}. Using INFO level."
            LOG_LEVEL=${LOG_LEVEL_INFO}
            ;;
    esac
}

# Initialize logging
init_logging() {
    # Set log level from environment
    if [ -n "${LOG_LEVEL:-}" ]; then
        set_log_level "${LOG_LEVEL}"
    fi

    # Log environment information
    log_environment
    log_aws_info
}
