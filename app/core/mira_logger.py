"""
Centralized logging configuration and wrapper for logging library.
"""

import logging
import sys
from typing import Optional

from app.core.config import settings


class ColoredFormatter(logging.Formatter):
    """Custom formatter that adds colors to log levels."""

    COLORS = {
        "INFO": "\033[92m",  # Green
        "ERROR": "\033[91m",  # Red
        "WARNING": "\033[93m",  # Yellow
        "DEBUG": "\033[94m",  # Blue
        "CRITICAL": "\033[95m",  # Magenta
        "EXCEPTION": "\033[96m",  # Orange
    }
    RESET = "\033[0m"

    def format(self, record):
        # Get the original formatted message
        log_message = super().format(record)

        # Add color to the level name
        level_name = record.levelname
        color = self.COLORS.get(level_name, "")
        reset = self.RESET

        # Replace the level name with colored version
        colored_level = f"{color}{level_name}{reset}"
        return log_message.replace(level_name, colored_level)


class MiraLogger:
    """Self-configuring logger with colored output."""

    _initialized = False
    _default_logger: Optional[logging.Logger] = None

    @classmethod
    def _get_default_logger(cls):
        """Get or create the default logger for static methods."""
        if cls._default_logger is None:
            cls._default_logger = logging.getLogger("mira")

            if not MiraLogger._initialized:
                cls._configure_root_logger()
                MiraLogger._initialized = True

            if not cls._default_logger.handlers:
                cls._configure_logger()
        return cls._default_logger

    @classmethod
    def _configure_root_logger(cls):
        """Configure the root logger with basic settings."""
        level = getattr(logging, settings.log_level.upper())
        logging.basicConfig(level=level)

        # Set specific logger levels
        logging.getLogger("uvicorn").setLevel(logging.INFO)
        logging.getLogger("fastapi").setLevel(logging.INFO)

        # Configure FastAPI logger to use our custom formatter
        fastapi_logger = logging.getLogger("fastapi")
        if not any(
            isinstance(h, logging.StreamHandler) for h in fastapi_logger.handlers
        ):
            handler = logging.StreamHandler(sys.stdout)
            handler.setFormatter(ColoredFormatter("%(levelname)s:     %(message)s"))
            fastapi_logger.addHandler(handler)
            fastapi_logger.propagate = False

    @classmethod
    def _configure_logger(cls):
        """Configure the default logger instance."""
        if cls._default_logger is None:
            return

        # Prevent propagation to root logger to avoid duplicate messages
        cls._default_logger.propagate = False

        handler = logging.StreamHandler(sys.stdout)
        handler.setFormatter(ColoredFormatter("%(levelname)s:     %(message)s"))
        cls._default_logger.addHandler(handler)

    # Static methods for direct class usage
    @classmethod
    def info(cls, message: str) -> None:
        """Log an info message."""
        logger = cls._get_default_logger()
        logger.info(message)

    @classmethod
    def error(cls, message: str) -> None:
        """Log an error message."""
        logger = cls._get_default_logger()
        logger.error(message)

    @classmethod
    def warning(cls, message: str) -> None:
        """Log a warning message."""
        logger = cls._get_default_logger()
        logger.warning(message)

    @classmethod
    def debug(cls, message: str) -> None:
        """Log a debug message."""
        logger = cls._get_default_logger()
        logger.debug(message)

    @classmethod
    def critical(cls, message: str) -> None:
        """Log a critical message."""
        logger = cls._get_default_logger()
        logger.critical(message)

    @classmethod
    def exception(cls, message: str) -> None:
        """Log an exception message."""
        logger = cls._get_default_logger()
        logger.exception(message)

    @classmethod
    def log(cls, level: int, message: str) -> None:
        """Log a message with the specified log level."""
        logger = cls._get_default_logger()
        logger.log(level, message)

    @classmethod
    def get_fastapi_logger(cls):
        """Get the FastAPI logger configured with our formatter."""
        if not MiraLogger._initialized:
            cls._configure_root_logger()
            MiraLogger._initialized = True
        return logging.getLogger("fastapi")
