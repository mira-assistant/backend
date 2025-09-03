"""
Application configuration using Pydantic BaseSettings.
"""
from typing import Optional, List
from pydantic_settings import BaseSettings
from pydantic import Field
import os


class Settings(BaseSettings):
    """Application settings with environment variable support."""

    # Application
    app_name: str = "Mira Backend"
    app_version: str = "4.3.0"
    debug: bool = False

    # Database
    database_url: str = Field(
        default="sqlite:///./mira.db",
        env="DATABASE_URL"
    )

    # LM Studio Configuration
    lm_studio_url: str = Field(
        default="http://localhost:1234/v1",
        env="LM_STUDIO_URL"
    )
    lm_studio_api_key: str = Field(
        default="lm-studio",
        env="LM_STUDIO_API_KEY"
    )

    # CORS
    cors_origins: List[str] = Field(
        default=["*"],
        env="CORS_ORIGINS"
    )

    # Security
    secret_key: str = Field(
        default="your-secret-key-here",
        env="SECRET_KEY"
    )
    access_token_expire_minutes: int = Field(
        default=30,
        env="ACCESS_TOKEN_EXPIRE_MINUTES"
    )

    # Logging
    log_level: str = Field(
        default="INFO",
        env="LOG_LEVEL"
    )

    # Audio Processing
    sample_rate: int = Field(
        default=16000,
        env="SAMPLE_RATE"
    )

    # Context Processing
    conversation_gap_threshold: int = Field(
        default=300,
        env="CONVERSATION_GAP_THRESHOLD"
    )
    context_similarity_threshold: float = Field(
        default=0.7,
        env="CONTEXT_SIMILARITY_THRESHOLD"
    )

    class Config:
        env_file = ".env"
        case_sensitive = False


# Global settings instance
settings = Settings()
