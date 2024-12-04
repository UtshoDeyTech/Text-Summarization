import logging
import os
import seqlog
import socket
from datetime import datetime
from config import SEQ_URL, SEQ_API_KEY, LOG_APP_NAME, LOG_ENVIROMENT

class FolderNameFilter(logging.Filter):
    def filter(self, record):
        folder_name = os.path.basename(os.path.dirname(record.pathname))
        record.folder_and_filename = f"{folder_name}/{record.filename}"
        return True

def setup_logger():
    seqlog.log_to_seq(
        server_url=SEQ_URL,
        api_key=SEQ_API_KEY,
        level=logging.INFO,
        batch_size=1,
        auto_flush_timeout=1
    )
    
    # Get environment from config
    environment = LOG_ENVIROMENT 
    app_name = LOG_APP_NAME   

    seqlog.set_global_log_properties(
        app=app_name,
        server=socket.gethostname(),
        env=environment
    )
    
    logger = logging.getLogger("AskKen")
    logger.setLevel(logging.INFO)
    
    console_handler = logging.StreamHandler()
    console_formatter = logging.Formatter('%(levelname)s: %(folder_and_filename)s:%(lineno)d - %(message)s')
    console_handler.setFormatter(console_formatter)
    console_handler.addFilter(FolderNameFilter())
    logger.addHandler(console_handler)
    
    return logger

logger = setup_logger()

def log_api_request(endpoint, method, status_code=None, **kwargs):
    log_data = {
        "endpoint": endpoint,
        "method": method,
        "status_code": status_code,
        "timestamp": datetime.utcnow().isoformat(),
        **kwargs
    }
    logger.info(f"API Request: {method} {endpoint} | {str(log_data)}")

def log_error(message, **kwargs):
    log_data = {
        "timestamp": datetime.utcnow().isoformat(),
        **kwargs
    }
    logger.error(f"{message} | {str(kwargs)}")