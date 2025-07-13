import logging
import os
import seqlog
import socket
from datetime import datetime
from config import SEQ_URL, SEQ_API_KEY, LOG_APP_NAME, LOG_ENVIROMENT, DEVELOPMENT_MODE

class FolderNameFilter(logging.Filter):
    def filter(self, record):
        folder_name = os.path.basename(os.path.dirname(record.pathname))
        record.folder_and_filename = f"{folder_name}/{record.filename}"
        return True

def setup_logger():
    # Check if we're in development mode
    is_development = str(DEVELOPMENT_MODE).lower() in ('true', '1', 'yes', 'on')
    
    if is_development:
        # Development mode - use only console logging
        logger = logging.getLogger("AskKen")
        logger.setLevel(logging.INFO)
        
        # Clear any existing handlers
        logger.handlers.clear()
        
        # Add console handler
        console_handler = logging.StreamHandler()
        console_formatter = logging.Formatter(
            '%(levelname)s: %(folder_and_filename)s:%(lineno)d - %(message)s'
        )
        console_handler.setFormatter(console_formatter)
        console_handler.addFilter(FolderNameFilter())
        logger.addHandler(console_handler)
        
        print(f"Logger initialized in DEVELOPMENT mode - using console only")
        
    else:
        # Production mode - use Seq logging
        if not SEQ_URL:
            print("WARNING: SEQ_URL not configured, falling back to console logging")
            return setup_console_only_logger()
            
        try:
            seqlog.log_to_seq(
                server_url=SEQ_URL,
                api_key=SEQ_API_KEY,
                level=logging.INFO,
                batch_size=1,
                auto_flush_timeout=1
            )
            
            # Get environment from config
            environment = LOG_ENVIROMENT or "production"
            app_name = LOG_APP_NAME or "pdf-processing-api"

            seqlog.set_global_log_properties(
                app=app_name,
                server=socket.gethostname(),
                env=environment
            )
            
            logger = logging.getLogger("AskKen")
            logger.setLevel(logging.INFO)
            
            # Also add console handler for production
            console_handler = logging.StreamHandler()
            console_formatter = logging.Formatter('%(levelname)s: %(folder_and_filename)s:%(lineno)d - %(message)s')
            console_handler.setFormatter(console_formatter)
            console_handler.addFilter(FolderNameFilter())
            logger.addHandler(console_handler)
            
            print(f"Logger initialized in PRODUCTION mode - using Seq at {SEQ_URL}")
            
        except Exception as e:
            print(f"Failed to initialize Seq logging: {e}")
            print("Falling back to console logging")
            return setup_console_only_logger()
    
    return logger

def setup_console_only_logger():
    """Fallback logger that only uses console output"""
    logger = logging.getLogger("AskKen")
    logger.setLevel(logging.INFO)
    
    # Clear any existing handlers
    logger.handlers.clear()
    
    # Add console handler
    console_handler = logging.StreamHandler()
    console_formatter = logging.Formatter(
        '%(levelname)s: %(folder_and_filename)s:%(lineno)d - %(message)s'
    )
    console_handler.setFormatter(console_formatter)
    console_handler.addFilter(FolderNameFilter())
    logger.addHandler(console_handler)
    
    return logger

def log_api_request(endpoint, method, status_code=None, **kwargs):
    log_data = {
        "endpoint": endpoint,
        "method": method,
        "status_code": status_code,
        "timestamp": datetime.utcnow().isoformat(),
        **kwargs
    }
    logger.info(f"API Request: {method} {endpoint}", extra=log_data)

def log_error(message, **kwargs):
    log_data = {
        "timestamp": datetime.utcnow().isoformat(),
        **kwargs
    }
    logger.error(message, extra=log_data)

# Initialize the logger
logger = setup_logger()