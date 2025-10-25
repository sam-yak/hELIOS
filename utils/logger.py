import logging
import sys
from datetime import datetime
from pathlib import Path

def setup_logger(name: str = "helios", log_file: str = "helios.log", level=logging.INFO):
    """
    Sets up a logger with both file and console handlers.
    
    Args:
        name: Logger name
        log_file: Path to log file
        level: Logging level (default: INFO)
    
    Returns:
        Configured logger instance
    """
    
    # Create logs directory if it doesn't exist
    log_dir = Path("logs")
    log_dir.mkdir(exist_ok=True)
    
    # Full path to log file
    log_path = log_dir / log_file
    
    # Create logger
    logger = logging.getLogger(name)
    logger.setLevel(level)
    
    # Avoid duplicate handlers
    if logger.handlers:
        return logger
    
    # Create formatters
    detailed_formatter = logging.Formatter(
        '%(asctime)s | %(name)s | %(levelname)s | %(funcName)s:%(lineno)d | %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    console_formatter = logging.Formatter(
        '%(levelname)s | %(message)s'
    )
    
    # File handler (detailed logging)
    file_handler = logging.FileHandler(log_path)
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(detailed_formatter)
    
    # Console handler (less verbose)
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(console_formatter)
    
    # Add handlers
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    
    return logger


def log_query_metrics(logger, query: str, response_time: float, num_sources: int, success: bool):
    """
    Logs query performance metrics.
    
    Args:
        logger: Logger instance
        query: User query
        response_time: Time taken in seconds
        num_sources: Number of sources retrieved
        success: Whether query succeeded
    """
    
    status = "SUCCESS" if success else "FAILED"
    
    logger.info(
        f"Query Metrics | Status: {status} | "
        f"Response Time: {response_time:.2f}s | "
        f"Sources: {num_sources} | "
        f"Query: '{query[:100]}...'"
    )
    
    # Log to separate metrics file for analysis
    metrics_path = Path("logs") / "metrics.log"
    with open(metrics_path, "a") as f:
        timestamp = datetime.now().isoformat()
        f.write(f"{timestamp},{status},{response_time:.3f},{num_sources},{query[:200]}\n")
