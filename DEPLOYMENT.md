# Deployment Guide

This guide covers deploying the Mira Backend to AWS using ECS Fargate.

## Prerequisites

1. AWS CLI installed and configured
2. Docker installed
3. Terraform installed (for infrastructure setup)
4. Access to AWS account with appropriate permissions

## Infrastructure Setup

1. Initialize Terraform:
```bash
cd terraform
terraform init
```

2. Create a `terraform.tfvars` file:
```bash
aws_region = "us-east-1"
db_name = "mira"
db_username = "mira"
db_password = "your-secure-password"
```

3. Plan and apply infrastructure:
```bash
terraform plan
terraform apply
```

## Environment Variables

Set the following environment variables:

```bash
export AWS_REGION=us-east-1
export AWS_ACCOUNT_ID=your-account-id
export ECR_REGISTRY=your-account-id.dkr.ecr.us-east-1.amazonaws.com
export ECR_REPOSITORY=mira-backend
export ECS_CLUSTER=mira-cluster
export ECS_SERVICE=mira-backend-service
```

## Deployment

### Using GitHub Actions (Recommended)

1. Set up the following secrets in your GitHub repository:
   - `AWS_ACCESS_KEY_ID`
   - `AWS_SECRET_ACCESS_KEY`

2. Push to the `main` branch to trigger automatic deployment

### Manual Deployment

1. Build and push Docker image:
```bash
./scripts/deploy.sh
```

## Monitoring

- CloudWatch Logs: `/ecs/mira-backend`
- ECS Service: `mira-backend-service`
- Load Balancer: Check AWS Console for ALB URL

## Scaling

To scale the service:

```bash
aws ecs update-service \
    --cluster mira-cluster \
    --service mira-backend-service \
    --desired-count 3
```

## Troubleshooting

1. Check ECS service events in AWS Console
2. Review CloudWatch logs
3. Verify security group rules
4. Check RDS connectivity

## Security Considerations

1. Use AWS Secrets Manager for sensitive data
2. Enable VPC Flow Logs
3. Use least privilege IAM roles
4. Enable CloudTrail for audit logging
5. Regular security updates and patches
