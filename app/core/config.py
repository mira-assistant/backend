"""
Application configuration using Pydantic BaseSettings.
"""

from typing import List
from pydantic_settings import BaseSettings
from pydantic import Field


class Settings(BaseSettings):
    """Application settings with environment variable support."""

    # Application
    app_name: str = "Mira Backend"
    app_version: str = "4.3.0"
    debug: bool = False

    # Database
    database_url: str = Field(default="sqlite:///./mira.db")

    # LM Studio Configuration
    lm_studio_url: str = Field(default="http://localhost:1234/v1")
    lm_studio_api_key: str = Field(default="lm-studio")

    # CORS
    cors_origins: List[str] = Field(default=["*"])

    # Logging
    log_level: str = Field(default="INFO")

    # Audio Processing
    sample_rate: int = Field(default=16000)

    # Context Processing
    conversation_gap_threshold: int = Field(default=300)
    context_similarity_threshold: float = Field(default=0.7)

    class Config:
        env_file = ".env"
        case_sensitive = False


# Global settings instance
settings = Settings()