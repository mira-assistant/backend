"""
OAuth configuration and utilities.
"""

from typing import Dict, Optional
from datetime import datetime, timedelta
import uuid
from authlib.integrations.starlette_client import OAuth
from starlette.config import Config
from fastapi import HTTPException, status

from app.core.config import settings

# Initialize OAuth instance
config = Config('.env')  # This will load from environment variables if .env doesn't exist
oauth = OAuth(config)

# Configure OAuth providers
oauth.register(
    name='google',
    server_metadata_url='https://accounts.google.com/.well-known/openid-configuration',
    client_kwargs={'scope': 'openid email profile'},
)

oauth.register(
    name='github',
    api_base_url='https://api.github.com/',
    access_token_url='https://github.com/login/oauth/access_token',
    authorize_url='https://github.com/login/oauth/authorize',
    client_kwargs={'scope': 'read:user user:email'},
)

oauth.register(
    name='microsoft',
    server_metadata_url='https://login.microsoftonline.com/common/v2.0/.well-known/openid-configuration',
    client_kwargs={'scope': 'openid email profile'},
)

async def get_oauth_user_data(provider: str, token: str) -> Dict:
    """Get user data from OAuth provider."""
    if provider not in oauth.clients:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Unsupported OAuth provider: {provider}"
        )

    client = oauth.clients[provider]

    if provider == 'google':
        resp = await client.get('https://www.googleapis.com/oauth2/v3/userinfo', token=token)
        if resp.status_code != 200:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Failed to get user info from Google"
            )
        user_data = resp.json()
        return {
            'provider_user_id': user_data['sub'],
            'email': user_data['email'],
            'username': user_data.get('name', '').lower().replace(' ', '_'),
            'user_data': user_data
        }

    elif provider == 'github':
        resp = await client.get('user', token=token)
        if resp.status_code != 200:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Failed to get user info from GitHub"
            )
        user_data = resp.json()

        # Get email separately as it might not be in the user profile
        email_resp = await client.get('user/emails', token=token)
        if email_resp.status_code != 200:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Failed to get user email from GitHub"
            )
        emails = email_resp.json()
        primary_email = next((e['email'] for e in emails if e['primary']), emails[0]['email'])

        return {
            'provider_user_id': str(user_data['id']),
            'email': primary_email,
            'username': user_data['login'],
            'user_data': user_data
        }

    elif provider == 'microsoft':
        resp = await client.get('https://graph.microsoft.com/v1.0/me', token=token)
        if resp.status_code != 200:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Failed to get user info from Microsoft"
            )
        user_data = resp.json()
        return {
            'provider_user_id': user_data['id'],
            'email': user_data['mail'] or user_data['userPrincipalName'],
            'username': user_data.get('displayName', '').lower().replace(' ', '_'),
            'user_data': user_data
        }

    raise HTTPException(
        status_code=status.HTTP_400_BAD_REQUEST,
        detail=f"Unsupported OAuth provider: {provider}"
    )


def generate_client_id() -> str:
    """Generate a unique client identifier."""
    return str(uuid.uuid4())
