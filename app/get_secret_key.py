import os
from dotenv import load_dotenv
from app.service.log_client import logger

def get_secret(key: str):
    """Retrieve secret from environment variables."""
    try:
        # Load environment variables from .env file
        load_dotenv()
        logger.info(f"Loading secret | key={key}")
        
        # Get the value from environment variables
        value = os.getenv(key)
        
        if value is None:
            logger.error(f"Secret not found | key={key}")
            raise KeyError(f"Key '{key}' not found in .env file")
        
        # Log success without exposing any part of the secret value
        logger.info(f"Secret retrieved successfully | key={key}, value_length={len(value)}")
        return value
        
    except Exception as e:
        error_msg = f"Secret retrieval failed | key={key}, error_type={type(e).__name__}, error={str(e)}"
        logger.error(error_msg)
        raise