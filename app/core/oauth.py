"""
OAuth2 client utilities for Google and GitHub.
"""

from typing import Dict, Optional

from authlib.integrations.starlette_client import OAuth
from starlette.config import Config

from core.config import settings

# OAuth configuration
config = Config()
oauth = OAuth(config)

# Google OAuth2
oauth.register(
    name='google',
    client_id=settings.google_client_id,
    client_secret=settings.google_client_secret,
    client_kwargs={
        'scope': 'openid email profile'
    },
    server_metadata_url='https://accounts.google.com/.well-known/openid-configuration',
)

# GitHub OAuth2  
oauth.register(
    name='github',
    client_id=settings.github_client_id,
    client_secret=settings.github_client_secret,
    client_kwargs={
        'scope': 'user:email'
    },
    access_token_url='https://github.com/login/oauth/access_token',
    authorize_url='https://github.com/login/oauth/authorize',
    api_base_url='https://api.github.com/',
)


def extract_user_info_google(user_info: Dict) -> Dict[str, Optional[str]]:
    """Extract user information from Google OAuth response."""
    return {
        'email': user_info.get('email'),
        'google_id': user_info.get('sub'),
        'username': user_info.get('name'),
    }


def extract_user_info_github(user_info: Dict, email_info: Optional[Dict] = None) -> Dict[str, Optional[str]]:
    """Extract user information from GitHub OAuth response."""
    email = None
    if email_info and isinstance(email_info, list):
        # Find primary email
        for email_data in email_info:
            if email_data.get('primary'):
                email = email_data.get('email')
                break
    
    return {
        'email': email or user_info.get('email'),
        'github_id': str(user_info.get('id')),
        'username': user_info.get('login'),
    }