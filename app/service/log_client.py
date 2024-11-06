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
    # Get the logger
    logger = logging.getLogger("app_logger")
    
    # Return existing logger if it's already configured
    if logger.hasHandlers():
        return logger
    
    # Clear any existing handlers and prevent propagation
    logger.handlers.clear()
    logger.propagate = False
    
    # Set base logging level
    logger.setLevel(logging.INFO)

    # Configure Seq logging
    SEQ_URL = os.getenv("SEQ_URL")
    SEQ_API_KEY = os.getenv("SEQ_API_KEY")

    print(f"Debug - SEQ_URL: {SEQ_URL}")  # Debug print
    print(f"Debug - SEQ_API_KEY exists: {bool(SEQ_API_KEY)}")  # Debug print

    if SEQ_URL:
        try:
            # Add Seq handler
            seq_handler = seqlog.SeqHandler(
                server_url=SEQ_URL,
                api_key=SEQ_API_KEY,
                batch_size=1,
                auto_flush_timeout=1
            )
            seq_handler.setLevel(logging.INFO)
            logger.addHandler(seq_handler)

            # Set global properties
            seqlog.set_global_log_properties(
                app="FastAPI-PDF-Processor",
                env=os.getenv("ENVIRONMENT", "development"),
                server=os.getenv("HOSTNAME", "unknown")
            )
            
            print("Debug - Seq handler added successfully")  # Debug print
        except Exception as e:
            print(f"Debug - Error setting up Seq: {str(e)}")  # Debug print
            # Continue with console logging even if Seq setup fails

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
    try:
        log_data = {
            "endpoint": endpoint,
            "method": method,
            "status_code": status_code,
            "timestamp": datetime.utcnow().isoformat(),
            **kwargs
        }
        
        # Format for console
        console_message = f"API Request: {method} {endpoint}"
        if kwargs:
            console_message += f" | {str(log_data)}"
            
        # Log with both console and structured data for Seq
        logger.info(console_message, extra=log_data)
    except Exception as e:
        print(f"Debug - Error in log_api_request: {str(e)}")  # Debug print
        logger.error(f"Error in log_api_request: {str(e)}")

def log_error(message, **kwargs):
    """Helper function for logging errors with structured data"""
    try:
        log_data = {
            "timestamp": datetime.utcnow().isoformat(),
            **kwargs
        }
        
        # Format for console
        console_message = message
        if kwargs:
            console_message += f" | {str(kwargs)}"
            
        # Log with both console and structured data for Seq
        logger.error(console_message, extra=log_data)
    except Exception as e:
        print(f"Debug - Error in log_error: {str(e)}")  # Debug print
        logger.error(f"Error in log_error: {str(e)}")