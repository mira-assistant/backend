# AWS Lambda Deployment Guide for Mira Backend

This guide explains how to deploy your Mira Backend to AWS Lambda with different LLM backend options.

## 🚨 The LLM Challenge

**Problem:** Your current setup uses LM Studio locally, but AWS Lambda can't:
- Store large model files (7B+ parameters)
- Connect to localhost
- Handle long cold starts for model loading
- Provide enough memory for model inference

## 🎯 Solution Options

### Option 1: AWS Bedrock (Recommended for Production)

**Best for:** Production deployment, cost-effective, serverless

```bash
# 1. Enable Bedrock in AWS Console
# 2. Set environment variables in Lambda
LLM_BACKEND=bedrock
AWS_REGION=us-east-1
BEDROCK_MODEL_ID=anthropic.claude-3-sonnet-20240229-v1:0
```

**Pros:**
- ✅ Serverless, pay-per-use
- ✅ No model hosting needed
- ✅ Multiple model options
- ✅ Integrates with Lambda

**Cons:**
- ❌ Loses your fine-tuned models
- ❌ Different API than LM Studio

### Option 2: Keep Fine-tuned Models with SageMaker

**Best for:** When you need to keep your custom models

```bash
# 1. Create SageMaker endpoint
aws sagemaker create-endpoint-config \
    --endpoint-config-name mira-llama-config \
    --production-variants VariantName=primary,ModelName=your-model,InitialInstanceCount=1,InstanceType=ml.t3.medium

# 2. Deploy endpoint
aws sagemaker create-endpoint \
    --endpoint-name mira-llama-endpoint \
    --endpoint-config-name mira-llama-config
```

**Pros:**
- ✅ Keep your fine-tuned models
- ✅ Serverless inference
- ✅ Scales automatically

**Cons:**
- ❌ More complex setup
- ❌ Higher costs for always-on endpoints

### Option 3: External API Services

**Best for:** Quick deployment, multiple model options

```bash
# Set environment variables
LLM_BACKEND=openai
OPENAI_API_KEY=your-openai-key
OPENAI_MODEL=gpt-4

# Or use Anthropic
LLM_BACKEND=anthropic
ANTHROPIC_API_KEY=your-anthropic-key
ANTHROPIC_MODEL=claude-3-sonnet-20240229
```

## 🚀 Deployment Steps

### 1. Prepare Your Code

```bash
# Install AWS dependencies
pip install boto3

# Update requirements.txt
echo "boto3>=1.34.0" >> requirements.txt
```

### 2. Configure Environment Variables

Create a `.env.production` file:

```bash
# Database
DATABASE_URL=postgresql://username:password@your-rds-endpoint.region.rds.amazonaws.com:5432/mira_db

# LLM Backend
LLM_BACKEND=bedrock  # or openai, anthropic, lm_studio

# AWS Bedrock (if using Bedrock)
AWS_REGION=us-east-1
BEDROCK_MODEL_ID=anthropic.claude-3-sonnet-20240229-v1:0

# OpenAI (if using OpenAI)
OPENAI_API_KEY=your-openai-key
OPENAI_MODEL=gpt-4

# Anthropic (if using Anthropic)
ANTHROPIC_API_KEY=your-anthropic-key
ANTHROPIC_MODEL=claude-3-sonnet-20240229
```

### 3. Deploy to Lambda

```bash
# Using AWS SAM
sam build
sam deploy --guided

# Or using Serverless Framework
serverless deploy
```

### 4. Set Up RDS Database

```bash
# Create RDS PostgreSQL instance
aws rds create-db-instance \
    --db-instance-identifier mira-db \
    --db-instance-class db.t3.micro \
    --engine postgres \
    --master-username mira_user \
    --master-user-password your_secure_password \
    --allocated-storage 20

# Run migrations
alembic upgrade head
```

## 🔧 Lambda Configuration

### Memory and Timeout
```yaml
# serverless.yml
functions:
  api:
    handler: app.main.handler
    memorySize: 1024  # Increase for LLM calls
    timeout: 30       # Increase for LLM calls
    environment:
      LLM_BACKEND: ${env:LLM_BACKEND}
      DATABASE_URL: ${env:DATABASE_URL}
```

### IAM Permissions
```json
{
  "Version": "2012-10-17",
  "Statement": [
    {
      "Effect": "Allow",
      "Action": [
        "bedrock:InvokeModel",
        "bedrock:InvokeModelWithResponseStream"
      ],
      "Resource": "*"
    },
    {
      "Effect": "Allow",
      "Action": [
        "rds:DescribeDBInstances",
        "rds:Connect"
      ],
      "Resource": "*"
    }
  ]
}
```

## 🧪 Testing Your Deployment

### 1. Test Database Connection
```bash
# Test RDS connection
alembic current
```

### 2. Test LLM Backend
```bash
# Test with curl
curl -X POST https://your-lambda-url.amazonaws.com/test-llm \
  -H "Content-Type: application/json" \
  -d '{"text": "Hello, how are you?"}'
```

### 3. Monitor Performance
```bash
# Check CloudWatch logs
aws logs describe-log-groups --log-group-name-prefix /aws/lambda/mira
```

## 💰 Cost Comparison

### AWS Bedrock
- **Cost:** ~$0.003 per 1K input tokens, ~$0.015 per 1K output tokens
- **Best for:** Production, variable usage

### SageMaker Endpoint
- **Cost:** ~$0.05-0.20 per hour + inference costs
- **Best for:** High usage, custom models

### External APIs
- **OpenAI:** ~$0.03 per 1K input tokens, ~$0.06 per 1K output tokens
- **Anthropic:** ~$0.003 per 1K input tokens, ~$0.015 per 1K output tokens

## 🎯 My Recommendation

**For your use case, I recommend:**

1. **Start with AWS Bedrock** - Easiest deployment, cost-effective
2. **Keep LM Studio for development** - Use `LLM_BACKEND=lm_studio` locally
3. **Switch to Bedrock for production** - Use `LLM_BACKEND=bedrock` in Lambda

This gives you:
- ✅ Easy local development with your current setup
- ✅ Simple production deployment
- ✅ Cost-effective scaling
- ✅ No infrastructure management

## 🔄 Migration Path

1. **Phase 1:** Deploy with Bedrock (lose fine-tuned models temporarily)
2. **Phase 2:** If needed, migrate fine-tuned models to SageMaker
3. **Phase 3:** Optimize based on usage patterns

Your code is already set up to handle this transition with the `LLM_BACKEND` configuration!
