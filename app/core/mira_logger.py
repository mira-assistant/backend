"""
Centralized logging configuration and wrapper for logging library.
"""

import logging
import sys
from typing import Optional
from app.core.config import settings


class MiraLogger:
    """Self-configuring logger with colored output."""

    _initialized = False
    _default_logger: Optional[logging.Logger] = None

    COLORS = {
        "INFO": "\033[92m",  # Green
        "ERROR": "\033[91m",  # Red
        "WARNING": "\033[93m",  # Yellow
        "DEBUG": "\033[94m",  # Blue
        "CRITICAL": "\033[95m",  # Magenta
        "EXCEPTION": "\033[96m",  # Orange
    }
    RESET = "\033[0m"

    @classmethod
    def _get_default_logger(cls):
        """Get or create the default logger for static methods."""
        if cls._default_logger is None:
            cls._default_logger = logging.getLogger("MiraLogger")

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

    @classmethod
    def _configure_logger(cls):
        """Configure the default logger instance."""
        if cls._default_logger is None:
            return
        handler = logging.StreamHandler(sys.stdout)
        handler.setFormatter(
            logging.Formatter("%(levelname)s:\t  %(name)s:%(message)s")
        )
        cls._default_logger.addHandler(handler)

    @classmethod
    def _colorize(cls, level: str, message: str) -> str:
        """Add color to the log message."""
        color = cls.COLORS.get(level, "")
        return f"{color}{message}{cls.RESET}"

    # Static methods for direct class usage
    @classmethod
    def info(cls, message: str) -> None:
        """Log an info message."""
        logger = cls._get_default_logger()
        logger.info(cls._colorize("INFO", message))

    @classmethod
    def error(cls, message: str) -> None:
        """Log an error message."""
        logger = cls._get_default_logger()
        logger.error(cls._colorize("ERROR", message))

    @classmethod
    def warning(cls, message: str) -> None:
        """Log a warning message."""
        logger = cls._get_default_logger()
        logger.warning(cls._colorize("WARNING", message))

    @classmethod
    def debug(cls, message: str) -> None:
        """Log a debug message."""
        logger = cls._get_default_logger()
        logger.debug(cls._colorize("DEBUG", message))

    @classmethod
    def critical(cls, message: str) -> None:
        """Log a critical message."""
        logger = cls._get_default_logger()
        logger.critical(cls._colorize("CRITICAL", message))

    @classmethod
    def exception(cls, message: str) -> None:
        """Log an exception message."""
        logger = cls._get_default_logger()
        logger.exception(cls._colorize("EXCEPTION", message))

    @classmethod
    def log(cls, level: int, message: str) -> None:
        """Log a message with the specified log level."""
        logger = cls._get_default_logger()
        level_name = logging.getLevelName(level)
        logger.log(level, cls._colorize(level_name, message))