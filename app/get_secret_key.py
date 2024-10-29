import os
from dotenv import load_dotenv
from app.service.log_client import logger

def get_secret(key: str):
    try:
        # Load environment variables from .env file
        load_dotenv()
        logger.info(f"Loading environment variable: {key}")
        
        # Get the value from environment variables
        value = os.getenv(key)
        
        if value is None:
            logger.error(f"Key '{key}' not found in .env file")
            raise KeyError(f"Key '{key}' not found in .env file")
        
        logger.info(f"Successfully retrieved value for key: {key}")
        return value
        
    except Exception as e:
        logger.error(f"Error retrieving secret for key {key}: {str(e)}")
        raise