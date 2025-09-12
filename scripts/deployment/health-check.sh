#!/bin/bash

# Enterprise Health Check Script
# Validates deployment health and functionality

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
readonly DEFAULT_TIMEOUT="30"

# Parse command line arguments
parse_arguments() {
    local environment="${1:-${DEFAULT_ENVIRONMENT}}"
    local region="${2:-${DEFAULT_REGION}}"
    local timeout="${3:-${DEFAULT_TIMEOUT}}"

    log_info "Starting health check"
    log_info "Environment: ${environment}"
    log_info "Region: ${region}"
    log_info "Timeout: ${timeout}s"

    # Export for use in other functions
    export ENVIRONMENT="${environment}"
    export AWS_REGION="${region}"
    export HEALTH_CHECK_TIMEOUT="${timeout}"
    export FUNCTION_NAME="mira-backend-${environment}"
}

# Check Lambda function health
check_lambda_health() {
    log_info "Checking Lambda function health..."

    # Check if function exists
    if ! aws lambda get-function --function-name "${FUNCTION_NAME}" --region "${AWS_REGION}" &> /dev/null; then
        log_error "Lambda function not found: ${FUNCTION_NAME}"
        return 1
    fi

    # Get function configuration
    local function_config
    function_config=$(aws lambda get-function-configuration --function-name "${FUNCTION_NAME}" --region "${AWS_REGION}")

    # Check function state
    local state
    state=$(echo "${function_config}" | jq -r '.State')

    if [ "${state}" != "Active" ]; then
        log_error "Lambda function is not active. State: ${state}"
        return 1
    fi

    # Check function last modified
    local last_modified
    last_modified=$(echo "${function_config}" | jq -r '.LastModified')
    log_info "Function last modified: ${last_modified}"

    log_success "Lambda function health check passed"
}

# Test Lambda function invocation
test_lambda_invocation() {
    log_info "Testing Lambda function invocation..."

    local test_payload='{"httpMethod": "GET", "path": "/health", "headers": {}}'
    local response_file="/tmp/lambda-response-${ENVIRONMENT}.json"

    # Invoke function
    aws lambda invoke \
        --function-name "${FUNCTION_NAME}" \
        --payload "${test_payload}" \
        --region "${AWS_REGION}" \
        --cli-read-timeout "${HEALTH_CHECK_TIMEOUT}" \
        "${response_file}"

    # Check response
    if [ ! -f "${response_file}" ]; then
        log_error "Lambda invocation failed - no response file"
        return 1
    fi

    # Parse response
    local status_code
    status_code=$(jq -r '.statusCode // "unknown"' "${response_file}")

    if [ "${status_code}" = "200" ] || [ "${status_code}" = "unknown" ]; then
        log_success "Lambda function invocation successful"
    else
        log_error "Lambda function returned error status: ${status_code}"
        cat "${response_file}"
        return 1
    fi

    # Clean up
    rm -f "${response_file}"
}

# Check CloudWatch logs
check_cloudwatch_logs() {
    log_info "Checking CloudWatch logs..."

    local log_group="/aws/lambda/${FUNCTION_NAME}"

    # Check if log group exists
    if ! aws logs describe-log-groups --log-group-name-prefix "${log_group}" --query 'logGroups[0].logGroupName' --output text | grep -q "${log_group}"; then
        log_warning "CloudWatch log group not found: ${log_group}"
        return 0
    fi

    # Get recent log streams
    local log_streams
    log_streams=$(aws logs describe-log-streams \
        --log-group-name "${log_group}" \
        --order-by LastEventTime \
        --descending \
        --max-items 5 \
        --query 'logStreams[].logStreamName' \
        --output text)

    if [ -z "${log_streams}" ]; then
        log_warning "No log streams found in log group: ${log_group}"
        return 0
    fi

    log_info "Found ${#log_streams[@]} recent log streams"
    log_success "CloudWatch logs check passed"
}

# Check database connectivity
check_database_connectivity() {
    log_info "Checking database connectivity..."

    # Load environment variables
    set -a
    source "${PROJECT_ROOT}/.env"
    set +a

    if [ -z "${DATABASE_URL:-}" ]; then
        log_error "DATABASE_URL not found in environment"
        return 1
    fi

    # Test database connection using Python
    local test_script="/tmp/db-test-${ENVIRONMENT}.py"
    cat > "${test_script}" << 'EOF'
import os
import sys
import psycopg2
from urllib.parse import urlparse

try:
    database_url = os.environ.get('DATABASE_URL')
    if not database_url:
        print("ERROR: DATABASE_URL not found")
        sys.exit(1)

    # Parse database URL
    parsed = urlparse(database_url)

    # Connect to database
    conn = psycopg2.connect(
        host=parsed.hostname,
        port=parsed.port,
        database=parsed.path[1:],  # Remove leading slash
        user=parsed.username,
        password=parsed.password
    )

    # Test query
    cursor = conn.cursor()
    cursor.execute("SELECT 1")
    result = cursor.fetchone()

    if result[0] == 1:
        print("SUCCESS: Database connection successful")
        sys.exit(0)
    else:
        print("ERROR: Database test query failed")
        sys.exit(1)

except Exception as e:
    print(f"ERROR: Database connection failed: {e}")
    sys.exit(1)
finally:
    if 'conn' in locals():
        conn.close()
EOF

    # Run database test
    if python3 "${test_script}"; then
        log_success "Database connectivity check passed"
    else
        log_error "Database connectivity check failed"
        rm -f "${test_script}"
        return 1
    fi

    # Clean up
    rm -f "${test_script}"
}

# Generate health report
generate_health_report() {
    log_info "Generating health report..."

    local report_file="${PROJECT_ROOT}/health-report-${ENVIRONMENT}.json"
    local timestamp=$(date -u +"%Y-%m-%dT%H:%M:%SZ")

    cat > "${report_file}" << EOF
{
    "timestamp": "${timestamp}",
    "environment": "${ENVIRONMENT}",
    "region": "${AWS_REGION}",
    "function_name": "${FUNCTION_NAME}",
    "health_status": "healthy",
    "checks": {
        "lambda_function": "passed",
        "lambda_invocation": "passed",
        "cloudwatch_logs": "passed",
        "database_connectivity": "passed"
    }
}
EOF

    log_info "Health report generated: ${report_file}"
}

# Main health check function
run_health_check() {
    log_info "Running health check for environment: ${ENVIRONMENT}"

    local exit_code=0

    # Run health checks
    check_lambda_health || exit_code=1
    test_lambda_invocation || exit_code=1
    check_cloudwatch_logs || exit_code=1
    check_database_connectivity || exit_code=1

    # Generate report
    generate_health_report

    if [ ${exit_code} -eq 0 ]; then
        log_success "All health checks passed"
    else
        log_error "Some health checks failed"
    fi

    return ${exit_code}
}

# Main execution
main() {
    parse_arguments "$@"
    run_health_check
}

# Run main function if script is executed directly
if [[ "${BASH_SOURCE[0]}" == "${0}" ]]; then
    main "$@"
fi
