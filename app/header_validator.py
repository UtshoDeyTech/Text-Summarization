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
        logger.info(
            f"Verifying headers | "
            f"has_authorization={bool(authorization and authorization.strip())}, "
            f"has_accept={bool(accept and accept.strip())}"
        )
        
        # Check Authorization header
        if not authorization or authorization.strip() == "":
            logger.error("Header verification failed | header=Authorization, reason=missing_or_empty")
            raise HTTPException(
                status_code=401,
                detail={
                    "status_code": "401",
                    "error_messages": ["Authorization header is required"]
                }
            )
        
        # Check Accept header
        if not accept or accept.strip() == "":
            logger.error("Header verification failed | header=Accept, reason=missing_or_empty")
            raise HTTPException(
                status_code=400,
                detail={
                    "status_code": "400",
                    "error_messages": ["Accept header is required"]
                }
            )
        
        # Log successful verification
        logger.info(
            f"Header verification successful | "
            f"auth_length={len(authorization)}, "
            f"accept_value={accept}"
        )
        return authorization, accept
        
    except HTTPException:
        raise
    except Exception as e:
        error_msg = (
            f"Header verification failed | "
            f"error_type={type(e).__name__}, "
            f"error={str(e)}"
        )
        logger.error(error_msg)
        raise HTTPException(
            status_code=500,
            detail={
                "status_code": "500",
                "error_messages": ["Internal server error during header verification"]
            }
        )