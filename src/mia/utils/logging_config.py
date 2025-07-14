"""
Structured logging configuration for M.I.A.
"""
import structlog
import logging
import sys
from pathlib import Path


def configure_logging(level="INFO", log_file="mia.log"):
    """Configure structured logging for the application."""
    
    # Create logs directory if it doesn't exist
    log_dir = Path("logs")
    log_dir.mkdir(exist_ok=True)
    
    # Configure structlog
    structlog.configure(
        processors=[
            structlog.contextvars.merge_contextvars,
            structlog.processors.add_log_level,
            structlog.processors.TimeStamper(fmt="ISO"),
            structlog.dev.ConsoleRenderer() if sys.stderr.isatty() else structlog.processors.JSONRenderer()
        ],
        wrapper_class=structlog.make_filtering_bound_logger(
            getattr(logging, level.upper())
        ),
        logger_factory=structlog.WriteLoggerFactory(),
        cache_logger_on_first_use=True,
    )
    
    # Configure standard logging
    logging.basicConfig(
        level=getattr(logging, level.upper()),
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        handlers=[
            logging.FileHandler(log_dir / log_file),
            logging.StreamHandler(sys.stdout)
        ]
    )
    
    return structlog.get_logger()


# Example usage:
# logger = configure_logging("INFO")
# logger.info("Application started", component="main", version="0.1.0")
