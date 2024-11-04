import logging
import os
from datetime import datetime
from dotenv import load_dotenv

load_dotenv()

class FolderNameFilter(logging.Filter):
    def filter(self, record):
        folder_name = os.path.basename(os.path.dirname(record.pathname))
        record.folder_and_filename = f"{folder_name}/{record.filename}"
        return True

def setup_logger():
    # Get the logger
    logger = logging.getLogger("app_logger")  # Use a specific name instead of __name__
    
    # Return existing logger if it's already configured
    if logger.hasHandlers():
        return logger
    
    # Clear any existing handlers and prevent propagation
    logger.handlers.clear()
    logger.propagate = False
    
    # Set base logging level
    logger.setLevel(logging.INFO)

    # Add custom formatter for console output
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_formatter = logging.Formatter(
        '%(levelname)s: %(folder_and_filename)s:%(lineno)d - %(message)s'
    )
    console_handler.setFormatter(console_formatter)
    
    # Add folder filter
    folder_filter = FolderNameFilter()
    console_handler.addFilter(folder_filter)
    
    # Add console handler to logger
    logger.addHandler(console_handler)

    return logger

# Create a global logger variable
logger = setup_logger()

def log_api_request(endpoint, method, status_code=None, **kwargs):
    """Helper function for logging API requests with structured data"""
    log_parts = [f"API Request: {method} {endpoint}"]
    if kwargs:
        log_parts.append(str(kwargs))
    logger.info(" | ".join(log_parts))

def log_error(message, **kwargs):
    """Helper function for logging errors with structured data"""
    if kwargs:
        message = f"{message} | {str(kwargs)}"
    logger.error(message)