from fastapi import HTTPException
from typing import Tuple
from app.service.log_client import logger

async def verify_headers(
    authorization: str,
    accept: str
) -> Tuple[str, str]:
    """
    Verify that required headers are present.
    
    Args:
        authorization (str): The Authorization header value
        accept (str): The Accept header value
        
    Returns:
        Tuple[str, str]: A tuple containing the validated (authorization, accept) headers
    """
    try:
        # Just check if headers are present and not empty
        if not authorization or authorization.strip() == "":
            logger.error("Missing Authorization header")
            raise HTTPException(
                status_code=401,
                detail={
                    "status_code": "401",
                    "error_messages": ["Authorization header is required"]
                }
            )
        
        if not accept or accept.strip() == "":
            logger.error("Missing Accept header")
            raise HTTPException(
                status_code=400,
                detail={
                    "status_code": "400",
                    "error_messages": ["Accept header is required"]
                }
            )
        
        # Log successful header verification
        logger.info("Headers verified successfully")
        return authorization, accept
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Unexpected error during header verification: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail={
                "status_code": "500",
                "error_messages": ["Internal server error during header verification"]
            }
        )