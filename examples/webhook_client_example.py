"""
Example client demonstrating multi-tenant backend API with webhooks.

This example shows how to:
1. Register a user and get JWT tokens
2. Register a client with a webhook URL
3. Receive webhook notifications when interactions are created

Usage:
    python webhook_client_example.py
"""

import asyncio
from typing import Optional

import httpx


# Configuration
BACKEND_URL = "http://localhost:8000/api/v2"


class BackendClient:
    """Client for interacting with the multi-tenant backend API."""
    
    def __init__(self, base_url: str):
        self.base_url = base_url
        self.access_token: Optional[str] = None
        self.refresh_token: Optional[str] = None
        self.user_id: Optional[str] = None
    
    async def register_user(self, email: str, password: str, username: str = None):
        """Register a new user and obtain tokens."""
        async with httpx.AsyncClient() as client:
            response = await client.post(
                f"{self.base_url}/auth/register",
                json={
                    "email": email,
                    "password": password,
                    "username": username or email.split("@")[0],
                },
            )
            response.raise_for_status()
            data = response.json()
            
            self.access_token = data["access_token"]
            self.refresh_token = data["refresh_token"]
            self.user_id = data["user"]["id"]
            
            print(f"✓ User registered: {data['user']['email']}")
            print(f"✓ User ID (network_id): {self.user_id}")
            return data
    
    def _get_headers(self) -> dict:
        """Get authentication headers."""
        if not self.access_token:
            raise ValueError("Not authenticated. Call register_user() first.")
        return {"Authorization": f"Bearer {self.access_token}"}
    
    async def register_client(self, client_id: str, webhook_url: str):
        """Register a client with webhook URL."""
        async with httpx.AsyncClient() as client:
            response = await client.post(
                f"{self.base_url}/service/client/register/{client_id}",
                headers=self._get_headers(),
                json={"webhook_url": webhook_url},
            )
            response.raise_for_status()
            data = response.json()
            
            print(f"✓ Client registered: {client_id}")
            print(f"  Webhook URL: {webhook_url}")
            return data


async def main():
    """Run example."""
    print("\n" + "="*60)
    print("Multi-Tenant Backend API - Example")
    print("="*60 + "\n")
    
    client = BackendClient(BACKEND_URL)
    
    # Register user
    await client.register_user(
        email="demo@example.com",
        password="demopassword123",
        username="demo_user",
    )
    
    # Register client with webhook
    await client.register_client(
        "demo-client",
        "https://webhook.site/unique-id"  # Replace with your webhook URL
    )
    
    print("\n✓ Setup complete!")


if __name__ == "__main__":
    asyncio.run(main())
