"""
Application configuration using Pydantic BaseSettings,
supports both local .env files and AWS Secrets Manager.
"""

import os
import json
from typing import List, Union

import boto3
from pydantic import Field, field_validator
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    """Application settings with environment variable support."""

    # Application
    app_name: str = "Mira API"
    app_version: str = "6.0.1"
    debug: bool = False

    # Database
    database_url: str = Field(
        default="postgresql://username:password@localhost:5432/mira"
    )

    # AWS Configuration
    aws_region: str = Field(default="us-east-1")
    env: str = Field(default="development")  # "development" or "production"

    # CORS
    cors_origins: Union[str, List[str]] = Field(default=["*"])

    @field_validator("cors_origins", mode="after")
    @classmethod
    def parse_cors_origins(cls, v):
        """Parse CORS origins from string or list."""
        if isinstance(v, str):
            return [v]
        return v

    # Logging
    log_level: str = Field(default="INFO")

    # API Keys
    gemini_api_key: str = Field(default="")
    huggingface_api_key: str = Field(default="")

    # Authentication
    secret_key: str = Field(default="your-secret-key-here-change-in-production")
    google_client_id: str = Field(default="")
    google_client_secret: str = Field(default="")
    github_client_id: str = Field(default="")
    github_client_secret: str = Field(default="")

    class Config:
        env_file = ".env"  # only used in development
        case_sensitive = False
        extra = "ignore"

    def load_aws_secrets(self, secret_name: str):
        """
        Load secrets from AWS Secrets Manager and override settings.
        """

        client = boto3.client("secretsmanager", region_name=self.aws_region)
        try:
            secret_response = client.get_secret_value(SecretId=secret_name)
            secret_string = secret_response.get("SecretString")
            if not secret_string:
                print(f"No secret string found for {secret_name}")
                return
            secret_data = json.loads(secret_string)
            for key, value in secret_data.items():
                # Only override existing fields
                if hasattr(self, key):
                    setattr(self, key, value)
            print(f"AWS secrets loaded from {secret_name}")
        except Exception as e:
            print(f"Error loading secrets from AWS: {e}")


# Global settings instance
settings = Settings()

# Automatically load AWS secrets if in production
try:
    settings.load_aws_secrets("mira-secrets")
except Exception as e:
    print(f"Failed to load AWS secrets: {e}")