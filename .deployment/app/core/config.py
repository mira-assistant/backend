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
    bedrock_model_id: str = Field(default="anthropic.claude-3-sonnet-20240229-v1:0")

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

    class Config:
        env_file = ".env"
        case_sensitive = False
        extra = "ignore"  # Allow extra fields from environment


# Global settings instance
settings = Settings()
