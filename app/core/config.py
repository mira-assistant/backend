"""
Application configuration using Pydantic BaseSettings.
"""

from typing import List, Union

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    """Application settings with environment variable support."""

    # Application
    app_name: str = "Mira Backend"
    app_version: str = "5.0.0"
    debug: bool = False

    # Database
    database_url: str = Field(
        default="postgresql://username:password@localhost:5432/mira"
    )

    # AWS Bedrock Configuration (for production)
    aws_region: str = Field(default="us-east-1")

    # CORS
    cors_origins: Union[str, List[str]] = Field(default=["*"])

    @field_validator("cors_origins", mode="after")
    @classmethod
    def parse_cors_origins(cls, v):
        """Parse CORS origins from string or list."""
        if isinstance(v, str):
            # If it's a single string, wrap it in a list
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
        extra = "ignore"  # Allow extra fields from environment


# Global settings instance
settings = Settings()
