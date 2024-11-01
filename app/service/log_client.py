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
    SEQ_API_KEY = os.getenv("SEQ_URL", None)

    # Configure Seq logging
    seqlog.log_to_seq(
        server_url=SEQ_URL,
        api_key=SEQ_API_KEY,
        level=logging.INFO,
        batch_size=1,  # Send events immediately
        auto_flush_timeout=1,  # Flush every second
        override_root_logger=True
    )

    # Get the logger
    logger = logging.getLogger(__name__)

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

    # Add structured logging properties
    seqlog.set_global_log_properties(
        app="FastAPI-PDF-Processor",
        env=os.getenv("ENVIRONMENT", "development"),
        server=os.getenv("HOSTNAME", "unknown")
    )

    return logger

# Create a global logger variable
logger = setup_logger()

# Helper functions for structured logging
def log_api_request(endpoint, method, status_code=None, **kwargs):
    """Helper function for logging API requests with structured data"""
    logger.info(
        f"API Request: {method} {endpoint}",
        endpoint=endpoint,
        method=method,
        status_code=status_code,
        timestamp=datetime.utcnow().isoformat(),
        **kwargs
    )

def log_error(message, **kwargs):
    """Helper function for logging errors with structured data"""
    logger.error(
        message,
        timestamp=datetime.utcnow().isoformat(),
        **kwargs
    )