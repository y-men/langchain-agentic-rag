"""
Simple logging configuration for UnderRadar.
"""
import logging
from typing import Optional


# Color codes for better logging
class Colors:
    PURPLE = "\033[95m"
    CYAN = "\033[96m"
    DARKCYAN = "\033[36m"
    BLUE = "\033[94m"
    GREEN = "\033[92m"
    YELLOW = "\033[93m"
    RED = "\033[91m"
    BOLD = "\033[1m"
    UNDERLINE = "\033[4m"
    END = "\033[0m"

# Default format
LOG_FORMAT = f'{Colors.CYAN}%(asctime)s | %(levelname)-5s | %(message)s{Colors.END}'
DATE_FORMAT = '%Y-%m-%d %H:%M:%S'



def setup_logger(name: Optional[str] = None, level: int = logging.DEBUG) -> logging.Logger:
    """
    Create and configure a logger.

    Args:
        name: Logger name (usually __name__)
        level: Logging level (default: INFO)

    Returns:
        Configured logger instance
    """
    logger = logging.getLogger(name or 'underradar')
    logger.setLevel(level)

    # Don't add handlers if they're already configured
    if not logger.handlers:
        handler = logging.StreamHandler()
        formatter = logging.Formatter(LOG_FORMAT, DATE_FORMAT)
        handler.setFormatter(formatter)
        logger.addHandler(handler)

    return logger

# Create a default logger instance
logger = setup_logger()
