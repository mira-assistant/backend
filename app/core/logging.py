"""
Centralized logging configuration.
"""

import logging
import sys
from typing import Optional
from app.core.config import settings


class ColorFormatter(logging.Formatter):
    """Colored log formatter for better readability."""

    COLORS = {
        "INFO": "\033[92m",  # Green
        "ERROR": "\033[91m",  # Red
        "WARNING": "\033[93m",  # Yellow
        "DEBUG": "\033[94m",  # Blue
        "CRITICAL": "\033[95m",  # Magenta
    }
    RESET = "\033[0m"

    def format(self, record):
        color = self.COLORS.get(record.levelname, "")
        record.levelname = f"{color}{record.levelname}{self.RESET}"
        return super().format(record)


def setup_logging(log_level: Optional[str] = None) -> None:
    """Set up application logging."""
    level = log_level or settings.log_level

    # Create handler
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(
        ColorFormatter(fmt="%(levelname)s:\t  %(name)s:%(message)s", datefmt="%Y-%m-%d %H:%M:%S")
    )

    # Configure root logger
    logging.basicConfig(
        level=getattr(logging, level.upper()),
        handlers=[handler],
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )

    # Set specific logger levels
    logging.getLogger("uvicorn").setLevel(logging.INFO)
    logging.getLogger("fastapi").setLevel(logging.INFO)


def get_logger(name: str) -> logging.Logger:
    """Get a logger instance."""
    return logging.getLogger(name)
