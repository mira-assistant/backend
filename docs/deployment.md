# 🚀 Deployment Guide

This guide covers deploying Mira Backend to AWS Lambda with RDS PostgreSQL database.

## 📋 Prerequisites

- AWS Account with appropriate permissions
- AWS CLI installed and configured
- Node.js (for Serverless Framework)
- Python 3.11+

## 🏗️ Architecture Overview

When deployed, Mira Backend creates:

- **AWS Lambda** - Serverless compute for your FastAPI app
- **API Gateway** - HTTP/WebSocket endpoints
- **RDS PostgreSQL** - Managed database
- **VPC** - Isolated network environment
- **IAM Roles** - Secure permissions

## 🚀 Quick Deployment

### 1. Setup Environment

```bash
# Setup AWS Lambda environment
./scripts/setup-lambda.sh

# Configure AWS credentials
aws configure
```

### 2. Configure Credentials

Create `.env` file with your API keys:

```bash
# Essential credentials
GEMINI_API_KEY=your_gemini_key_here
DATABASE_URL=sqlite:///./mira.db
DB_PASSWORD=MiraDB123!

# LLM Backend selection
LLM_BACKEND=gemini
```

### 3. Deploy to AWS

```bash
# Deploy everything
./scripts/deploy.sh

# Or deploy to specific stage
./scripts/deploy.sh -s prod -r us-west-2
```

### 4. Run Database Migrations

```bash
# Migrate database schema
./scripts/migrate-db.sh
```

## 🔧 Configuration

### Environment Variables

| Variable | Description | Default |
|----------|-------------|---------|
| `GEMINI_API_KEY` | Google Gemini API key | Required |
| `OPENAI_API_KEY` | OpenAI API key | Optional |
| `LLM_BACKEND` | LLM provider | `gemini` |
| `DATABASE_URL` | Database connection | `sqlite:///./mira.db` |
| `DB_PASSWORD` | RDS password | `MiraDB123!` |

### AWS Resources Created

#### Lambda Function
- **Runtime**: Python 3.11
- **Memory**: 1024 MB
- **Timeout**: 30 seconds
- **Environment**: All your `.env` variables

#### RDS Database
- **Engine**: PostgreSQL
- **Instance**: db.t3.micro
- **Storage**: 20 GB
- **Backup**: 7 days retention

#### VPC Configuration
- **VPC**: Isolated network
- **Subnets**: 2 public, 2 private
- **Security Groups**: Lambda ↔ RDS only

## 🛠️ Advanced Configuration

### Custom AWS Profile

```bash
# Use specific AWS profile
./scripts/deploy.sh -p my-aws-profile
```

### Different Stages

```bash
# Deploy to staging
./scripts/deploy.sh -s staging

# Deploy to production
./scripts/deploy.sh -s prod
```

### Custom Region

```bash
# Deploy to different region
./scripts/deploy.sh -r us-west-2
```

## 🔍 Monitoring & Debugging

### View Logs

```bash
# View Lambda logs
serverless logs -f api --stage dev

# Follow logs in real-time
serverless logs -f api --stage dev --tail
```

### Test Deployment

```bash
# Test API endpoint
curl https://your-api-gateway-url/

# Test WebSocket
wscat -c wss://your-api-gateway-url/ws/your-network-id
```

### Database Access

```bash
# Connect to RDS (if psql installed)
psql -h your-rds-endpoint -U mira_user -d mira_db
```

## 🗑️ Cleanup

### Remove Deployment

```bash
# Remove all AWS resources
serverless remove --stage dev

# Remove specific stage
serverless remove --stage prod
```

### Manual Cleanup

If automatic cleanup fails:

1. Delete Lambda function
2. Delete API Gateway
3. Delete RDS instance
4. Delete VPC and subnets
5. Delete IAM roles

## 🚨 Troubleshooting

### Common Issues

#### 1. AWS Credentials Not Found
```bash
# Configure AWS CLI
aws configure

# Or set environment variables
export AWS_ACCESS_KEY_ID=your_key
export AWS_SECRET_ACCESS_KEY=your_secret
```

#### 2. Database Connection Failed
```bash
# Check RDS endpoint
aws rds describe-db-instances

# Verify security groups
aws ec2 describe-security-groups
```

#### 3. Lambda Timeout
- Increase timeout in `serverless.yml`
- Check CloudWatch logs for errors
- Verify database connection

#### 4. API Gateway 502 Error
- Check Lambda function logs
- Verify environment variables
- Test function directly in AWS Console

### Debug Commands

```bash
# Check AWS credentials
aws sts get-caller-identity

# List Lambda functions
aws lambda list-functions

# Check RDS status
aws rds describe-db-instances --query 'DBInstances[0].DBInstanceStatus'
```

## 📊 Cost Optimization

### Lambda
- Use appropriate memory size
- Optimize cold start times
- Consider provisioned concurrency for production

### RDS
- Use appropriate instance size
- Enable automated backups
- Monitor storage usage

### API Gateway
- Use HTTP API (cheaper than REST API)
- Implement caching where possible
- Monitor request volume

## 🔒 Security Best Practices

1. **Use IAM Roles** - Never hardcode AWS credentials
2. **Encrypt Secrets** - Use AWS Secrets Manager for sensitive data
3. **VPC Security** - Keep database in private subnets
4. **API Keys** - Rotate API keys regularly
5. **Monitoring** - Enable CloudTrail and CloudWatch

## 📈 Scaling

### Horizontal Scaling
- Lambda automatically scales
- RDS can be scaled up/down
- API Gateway handles high traffic

### Vertical Scaling
- Increase Lambda memory
- Upgrade RDS instance class
- Add read replicas for database

---

**Need help?** Check the [main README](../README.md) or open an issue.
