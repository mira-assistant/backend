# Multi-Tenant Backend API Examples

This directory contains example code demonstrating how to use the multi-tenant backend API with JWT authentication and webhooks.

## webhook_client_example.py

A simple Python example showing how to:
- Register a user and obtain JWT tokens
- Register a client with a webhook URL for receiving interaction updates

### Prerequisites

```bash
pip install httpx
```

### Usage

```bash
python webhook_client_example.py
```

### What it demonstrates

1. **User Registration**: Creates a user account and receives JWT tokens
2. **Client Registration**: Registers a client with a webhook URL
3. **Authentication**: Shows how to use JWT tokens in API requests

## Testing Webhooks

To test webhook functionality:

1. Get a test webhook URL from [webhook.site](https://webhook.site)
2. Update the webhook URL in the example
3. Run the example to register your client
4. Create an interaction using the API
5. Check webhook.site to see the received payload

## See Also

- [WEBHOOK_USAGE.md](../WEBHOOK_USAGE.md) - Comprehensive usage guide
- [API Documentation](http://localhost:8000/docs) - Interactive API docs (when server is running)
