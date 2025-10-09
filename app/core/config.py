"""
Application configuration using Pydantic BaseSettings,
supports both local .env files and AWS Secrets Manager.
"""

import json
from typing import List, Union

import boto3
from pydantic import Field, field_validator
from pydantic_settings import BaseSettings
import io
import os


class Settings(BaseSettings):
    """Application settings with environment variable support."""

    # Application
    app_name: str = "Mira API"
    app_version: str = "6.1.0"
    debug: bool = False

    # Database
    database_url: str = Field(default="")

    # AWS Configuration
    aws_region: str = Field(default="us-east-1")

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
        env_file = ".env"
        case_sensitive = False
        extra = "ignore"

    def load_aws_secrets(self, secret_name: str):
        try:
            client = boto3.client("secretsmanager", region_name=self.aws_region)
            secret_response = client.get_secret_value(SecretId=secret_name)
            secret_string = secret_response.get("SecretString")

            if not secret_string:
                print(f"No secret string found for {secret_name}")
                return

            # Try JSON parse first
            try:
                secret_data = json.loads(secret_string)
            except json.JSONDecodeError:
                # Fallback to .env format
                secret_data = {}
                for line in io.StringIO(secret_string):
                    if line.strip() and not line.startswith("#"):
                        key, value = line.strip().split("=", 1)
                        secret_data[key.strip()] = value.strip()

            for key, value in secret_data.items():
                if hasattr(self, key.lower()):
                    setattr(self, key.lower(), value)
                elif hasattr(self, key.upper()):
                    setattr(self, key.upper(), value)
                else:
                    os.environ[key] = value  # fallback

            print(f"AWS secrets loaded from {secret_name}")

        except Exception as e:
            print(f"Error loading secrets from AWS: {e}")


# Global settings instance
settings = Settings()
