"""
Logging configuration for Synapso Core.

This module provides centralized logging configuration and utilities
for consistent logging across the Synapso system.
"""

import logging

logging.getLogger("torch").setLevel(logging.WARNING)


def get_logger(name: str) -> logging.Logger:
    """
    Get a configured logger instance for the specified module.

    Args:
        name: The name of the module requesting the logger

    Returns:
        logging.Logger: A configured logger instance with appropriate
                       handlers and formatters
    """
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    handler = logging.StreamHandler()
    formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    return logger
