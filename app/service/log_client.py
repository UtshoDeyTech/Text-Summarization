import logging
import os
import seqlog
from datetime import datetime
from dotenv import load_dotenv

load_dotenv()

class FolderNameFilter(logging.Filter):
    def filter(self, record):
        folder_name = os.path.basename(os.path.dirname(record.pathname))
        record.folder_and_filename = f"{folder_name}/{record.filename}"
        return True

def setup_logger():
    # Seq configuration
    SEQ_URL = os.getenv("SEQ_URL")
    SEQ_API_KEY = os.getenv("SEQ_API_KEY", None)

    # Get the logger
    logger = logging.getLogger(__name__)
    
    # Clear any existing handlers
    logger.handlers.clear()
    
    # Prevent propagation to root logger
    logger.propagate = False
    
    # Set base logging level
    logger.setLevel(logging.INFO)

    # Configure Seq logging if URL is provided
    if SEQ_URL:
        seqlog.log_to_seq(
            server_url=SEQ_URL,
            api_key=SEQ_API_KEY,
            level=logging.INFO,
            batch_size=1,
            auto_flush_timeout=1,
            override_root_logger=False
        )

        # Add structured logging properties
        seqlog.set_global_log_properties(
            app="FastAPI-PDF-Processor",
            env=os.getenv("ENVIRONMENT", "development"),
            server=os.getenv("HOSTNAME", "unknown")
        )

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
    message = f"API Request: {method} {endpoint}"
    if kwargs:
        message += f" | {str(kwargs)}"
    logger.info(message)

def log_error(message, **kwargs):
    """Helper function for logging errors with structured data"""
    if kwargs:
        message += f" | {str(kwargs)}"
    logger.error(message)