#!/bin/bash

# Enterprise AWS Utilities
# Provides AWS-specific functionality and operations

# AWS utility functions
get_aws_account_id() {
    aws sts get-caller-identity --query Account --output text 2>/dev/null || echo "unknown"
}

get_aws_region() {
    aws configure get region 2>/dev/null || echo "us-east-1"
}

get_aws_profile() {
    aws configure get profile 2>/dev/null || echo "default"
}

check_aws_cli_version() {
    local required_version="${1:-2.0.0}"
    local current_version

    if ! command -v aws &> /dev/null; then
        log_error "AWS CLI is not installed"
        return 1
    fi

    current_version=$(aws --version 2>&1 | grep -oP '\d+\.\d+\.\d+' | head -n1)

    if [ -z "${current_version}" ]; then
        log_warning "Could not determine AWS CLI version"
        return 0
    fi

    log_info "AWS CLI version: ${current_version}"

    # Simple version comparison (basic)
    if [ "${current_version}" != "${required_version}" ]; then
        log_warning "AWS CLI version ${current_version} may not be optimal"
        log_info "Recommended version: ${required_version}"
    fi

    return 0
}

wait_for_lambda_function() {
    local function_name="$1"
    local region="$2"
    local max_attempts="${3:-30}"
    local wait_interval="${4:-10}"
    local attempt=0

    log_info "Waiting for Lambda function to be ready: ${function_name}"

    while [ ${attempt} -lt ${max_attempts} ]; do
        if aws lambda get-function --function-name "${function_name}" --region "${region}" &> /dev/null; then
            local state
            state=$(aws lambda get-function-configuration --function-name "${function_name}" --region "${region}" --query State --output text)

            if [ "${state}" = "Active" ]; then
                log_success "Lambda function is ready: ${function_name}"
                return 0
            fi

            log_info "Lambda function state: ${state}, waiting..."
        else
            log_info "Lambda function not found, waiting..."
        fi

        ((attempt++))
        sleep ${wait_interval}
    done

    log_error "Lambda function did not become ready within ${max_attempts} attempts"
    return 1
}

wait_for_api_gateway() {
    local api_id="$1"
    local region="$2"
    local max_attempts="${3:-20}"
    local wait_interval="${4:-15}"
    local attempt=0

    log_info "Waiting for API Gateway to be ready: ${api_id}"

    while [ ${attempt} -lt ${max_attempts} ]; do
        if aws apigatewayv2 get-api --api-id "${api_id}" --region "${region}" &> /dev/null; then
            log_success "API Gateway is ready: ${api_id}"
            return 0
        fi

        log_info "API Gateway not ready, waiting... (attempt ${attempt}/${max_attempts})"
        ((attempt++))
        sleep ${wait_interval}
    done

    log_error "API Gateway did not become ready within ${max_attempts} attempts"
    return 1
}

get_lambda_function_arn() {
    local function_name="$1"
    local region="$2"

    aws lambda get-function --function-name "${function_name}" --region "${region}" --query 'Configuration.FunctionArn' --output text 2>/dev/null || echo ""
}

get_api_gateway_url() {
    local api_id="$1"
    local region="$2"

    aws apigatewayv2 get-api --api-id "${api_id}" --region "${region}" --query 'ApiEndpoint' --output text 2>/dev/null || echo ""
}

create_cloudwatch_alarm() {
    local alarm_name="$1"
    local function_name="$2"
    local region="$3"
    local threshold="${4:-5}"

    log_info "Creating CloudWatch alarm: ${alarm_name}"

    aws cloudwatch put-metric-alarm \
        --alarm-name "${alarm_name}" \
        --alarm-description "Alarm for Lambda function errors: ${function_name}" \
        --metric-name Errors \
        --namespace AWS/Lambda \
        --statistic Sum \
        --period 300 \
        --threshold "${threshold}" \
        --comparison-operator GreaterThanThreshold \
        --dimensions Name=FunctionName,Value="${function_name}" \
        --evaluation-periods 2 \
        --region "${region}" \
        --treat-missing-data notBreaching || {
        log_warning "Failed to create CloudWatch alarm: ${alarm_name}"
        return 1
    }

    log_success "CloudWatch alarm created: ${alarm_name}"
    return 0
}

tag_lambda_function() {
    local function_name="$1"
    local region="$2"
    local environment="$3"
    local version="${4:-1.0.0}"

    log_info "Tagging Lambda function: ${function_name}"

    aws lambda tag-resource \
        --resource "${function_name}" \
        --tags "Environment=${environment},Version=${version},Project=MiraBackend,ManagedBy=CI/CD" \
        --region "${region}" || {
        log_warning "Failed to tag Lambda function: ${function_name}"
        return 1
    }

    log_success "Lambda function tagged: ${function_name}"
    return 0
}

get_lambda_function_logs() {
    local function_name="$1"
    local region="$2"
    local start_time="${3:-$(date -u -d '1 hour ago' +%s)}"
    local end_time="${4:-$(date -u +%s)}"

    local log_group="/aws/lambda/${function_name}"

    log_info "Retrieving logs for function: ${function_name}"

    aws logs filter-log-events \
        --log-group-name "${log_group}" \
        --start-time "${start_time}000" \
        --end-time "${end_time}000" \
        --region "${region}" \
        --query 'events[].message' \
        --output text 2>/dev/null || {
        log_warning "Failed to retrieve logs for function: ${function_name}"
        return 1
    }
}

check_lambda_function_errors() {
    local function_name="$1"
    local region="$2"
    local hours="${3:-1}"

    local start_time=$(date -u -d "${hours} hours ago" +%s)
    local end_time=$(date -u +%s)

    log_info "Checking for errors in Lambda function: ${function_name}"

    local error_count
    error_count=$(aws logs filter-log-events \
        --log-group-name "/aws/lambda/${function_name}" \
        --start-time "${start_time}000" \
        --end-time "${end_time}000" \
        --filter-pattern "ERROR" \
        --region "${region}" \
        --query 'events | length(@)' \
        --output text 2>/dev/null || echo "0")

    if [ "${error_count}" -gt 0 ]; then
        log_warning "Found ${error_count} errors in the last ${hours} hour(s)"
        return 1
    else
        log_success "No errors found in the last ${hours} hour(s)"
        return 0
    fi
}

create_lambda_function_url() {
    local function_name="$1"
    local region="$2"
    local auth_type="${3:-NONE}"

    log_info "Creating Lambda function URL: ${function_name}"

    local function_url
    function_url=$(aws lambda create-function-url-config \
        --function-name "${function_name}" \
        --auth-type "${auth_type}" \
        --region "${region}" \
        --query 'FunctionUrl' \
        --output text 2>/dev/null || echo "")

    if [ -n "${function_url}" ]; then
        log_success "Lambda function URL created: ${function_url}"
        echo "${function_url}"
    else
        log_warning "Failed to create Lambda function URL"
        return 1
    fi
}

get_lambda_function_metrics() {
    local function_name="$1"
    local region="$2"
    local start_time="${3:-$(date -u -d '1 hour ago' +%Y-%m-%dT%H:%M:%S)}"
    local end_time="${4:-$(date -u +%Y-%m-%dT%H:%M:%S)}"

    log_info "Retrieving metrics for function: ${function_name}"

    aws cloudwatch get-metric-statistics \
        --namespace AWS/Lambda \
        --metric-name Invocations \
        --dimensions Name=FunctionName,Value="${function_name}" \
        --start-time "${start_time}" \
        --end-time "${end_time}" \
        --period 300 \
        --statistics Sum \
        --region "${region}" \
        --query 'Datapoints[0].Sum' \
        --output text 2>/dev/null || echo "0"
}

cleanup_old_lambda_versions() {
    local function_name="$1"
    local region="$2"
    local keep_versions="${3:-5}"

    log_info "Cleaning up old Lambda versions: ${function_name}"

    local versions
    versions=$(aws lambda list-versions-by-function \
        --function-name "${function_name}" \
        --region "${region}" \
        --query 'Versions[?Version != `$LATEST`].Version' \
        --output text 2>/dev/null || echo "")

    if [ -z "${versions}" ]; then
        log_info "No old versions to clean up"
        return 0
    fi

    local version_count
    version_count=$(echo "${versions}" | wc -w)

    if [ "${version_count}" -le "${keep_versions}" ]; then
        log_info "Version count (${version_count}) is within limit (${keep_versions})"
        return 0
    fi

    local versions_to_delete
    versions_to_delete=$(echo "${versions}" | tr ' ' '\n' | sort -V | head -n -${keep_versions})

    for version in ${versions_to_delete}; do
        log_info "Deleting Lambda version: ${version}"
        aws lambda delete-function \
            --function-name "${function_name}" \
            --qualifier "${version}" \
            --region "${region}" || {
            log_warning "Failed to delete version: ${version}"
        }
    done

    log_success "Old Lambda versions cleaned up"
    return 0
}
