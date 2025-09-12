#!/bin/bash

# Enterprise Deployment Notification Script
# Sends notifications about deployment status

set -euo pipefail

# Script configuration
readonly SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
readonly PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
readonly SCRIPT_NAME="$(basename "$0")"

# Load utilities
source "${SCRIPT_DIR}/../utilities/logging.sh"
source "${SCRIPT_DIR}/../utilities/validation.sh"

# Default configuration
readonly DEFAULT_ENVIRONMENT="dev"

# Parse command line arguments
parse_arguments() {
    local status="$1"
    local environment="${2:-${DEFAULT_ENVIRONMENT}}"

    log_info "Sending deployment notification"
    log_info "Status: ${status}"
    log_info "Environment: ${environment}"

    # Export for use in other functions
    export DEPLOYMENT_STATUS="${status}"
    export ENVIRONMENT="${environment}"
}

# Send Slack notification (if configured)
send_slack_notification() {
    local webhook_url="${SLACK_WEBHOOK_URL:-}"

    if [ -z "${webhook_url}" ]; then
        log_info "Slack webhook not configured, skipping Slack notification"
        return 0
    fi

    log_info "Sending Slack notification..."

    local color
    local emoji
    case "${DEPLOYMENT_STATUS}" in
        "success")
            color="good"
            emoji="✅"
            ;;
        "failure")
            color="danger"
            emoji="❌"
            ;;
        *)
            color="warning"
            emoji="⚠️"
            ;;
    esac

    local message
    message=$(cat << EOF
{
    "text": "${emoji} Mira Backend Deployment ${DEPLOYMENT_STATUS^}",
    "attachments": [
        {
            "color": "${color}",
            "fields": [
                {
                    "title": "Environment",
                    "value": "${ENVIRONMENT}",
                    "short": true
                },
                {
                    "title": "Status",
                    "value": "${DEPLOYMENT_STATUS}",
                    "short": true
                },
                {
                    "title": "Timestamp",
                    "value": "$(date -u +"%Y-%m-%d %H:%M:%S UTC")",
                    "short": false
                }
            ]
        }
    ]
}
EOF
)

    if curl -X POST -H 'Content-type: application/json' \
        --data "${message}" \
        "${webhook_url}" &> /dev/null; then
        log_success "Slack notification sent"
    else
        log_warning "Failed to send Slack notification"
    fi
}

# Send email notification (if configured)
send_email_notification() {
    local smtp_server="${SMTP_SERVER:-}"
    local smtp_port="${SMTP_PORT:-587}"
    local smtp_user="${SMTP_USER:-}"
    local smtp_password="${SMTP_PASSWORD:-}"
    local email_to="${EMAIL_TO:-}"

    if [ -z "${smtp_server}" ] || [ -z "${email_to}" ]; then
        log_info "Email configuration not complete, skipping email notification"
        return 0
    fi

    log_info "Sending email notification..."

    local subject
    subject="Mira Backend Deployment ${DEPLOYMENT_STATUS^} - ${ENVIRONMENT}"

    local body
    body=$(cat << EOF
Mira Backend Deployment Notification

Environment: ${ENVIRONMENT}
Status: ${DEPLOYMENT_STATUS}
Timestamp: $(date -u +"%Y-%m-%d %H:%M:%S UTC")
Repository: ${GITHUB_REPOSITORY:-unknown}
Commit: ${GITHUB_SHA:-unknown}

This is an automated notification from the CI/CD pipeline.
EOF
)

    # Use mail command if available
    if command -v mail &> /dev/null; then
        echo "${body}" | mail -s "${subject}" "${email_to}" && {
            log_success "Email notification sent"
        } || {
            log_warning "Failed to send email notification"
        }
    else
        log_warning "Mail command not available, skipping email notification"
    fi
}

# Create deployment report
create_deployment_report() {
    local report_file="${PROJECT_ROOT}/deployment-report-${ENVIRONMENT}-$(date +%Y%m%d-%H%M%S).json"

    log_info "Creating deployment report: ${report_file}"

    local report
    report=$(cat << EOF
{
    "deployment": {
        "status": "${DEPLOYMENT_STATUS}",
        "environment": "${ENVIRONMENT}",
        "timestamp": "$(date -u +"%Y-%m-%dT%H:%M:%SZ")",
        "repository": "${GITHUB_REPOSITORY:-unknown}",
        "commit": "${GITHUB_SHA:-unknown}",
        "actor": "${GITHUB_ACTOR:-unknown}",
        "workflow": "${GITHUB_WORKFLOW:-unknown}"
    },
    "system": {
        "hostname": "$(hostname)",
        "os": "$(uname -s)",
        "architecture": "$(uname -m)"
    },
    "aws": {
        "region": "${AWS_REGION:-unknown}",
        "account_id": "$(aws sts get-caller-identity --query Account --output text 2>/dev/null || echo "unknown")"
    }
}
EOF
)

    echo "${report}" > "${report_file}"
    log_success "Deployment report created: ${report_file}"
}

# Update deployment status in external system (if configured)
update_external_status() {
    local status_endpoint="${STATUS_ENDPOINT:-}"

    if [ -z "${status_endpoint}" ]; then
        log_info "Status endpoint not configured, skipping external status update"
        return 0
    fi

    log_info "Updating external status system..."

    local status_data
    status_data=$(cat << EOF
{
    "service": "mira-backend",
    "environment": "${ENVIRONMENT}",
    "status": "${DEPLOYMENT_STATUS}",
    "timestamp": "$(date -u +"%Y-%m-%dT%H:%M:%SZ")"
}
EOF
)

    if curl -X POST -H 'Content-Type: application/json' \
        --data "${status_data}" \
        "${status_endpoint}" &> /dev/null; then
        log_success "External status updated"
    else
        log_warning "Failed to update external status"
    fi
}

# Main notification function
send_notifications() {
    log_info "Sending deployment notifications for environment: ${ENVIRONMENT}"

    send_slack_notification
    send_email_notification
    create_deployment_report
    update_external_status

    log_success "Deployment notifications completed"
}

# Main execution
main() {
    parse_arguments "$@"
    send_notifications
}

# Run main function if script is executed directly
if [[ "${BASH_SOURCE[0]}" == "${0}" ]]; then
    main "$@"
fi
