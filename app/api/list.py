import logging
from fastapi import APIRouter, HTTPException, Header, Path, Request
from fastapi.responses import JSONResponse
from app.service.s3_storage import list_objects, S3_BUCKET_NAME
from app.service.log_client import logger

router = APIRouter()

@router.get("/{user_id}/list_pdfs")
async def list_pdfs(
    request: Request,
    user_id: str
):
    try:
        logger.info(f"Starting PDF listing | user_id={user_id}, bucket={S3_BUCKET_NAME}")
        
        response = list_objects(S3_BUCKET_NAME)
        pdfs = []
        
        total_objects = len(response.get('Contents', []))
        logger.info(f"Processing S3 objects | user_id={user_id}, total_objects={total_objects}")
        
        for obj in response.get('Contents', []):
            if obj['Key'].endswith('.pdf'):
                path_parts = obj['Key'].split('/')
                if len(path_parts) > 1 and path_parts[0] == user_id:
                    pdf_id = path_parts[1].split('.')[0]
                    pdfs.append({
                        "id": pdf_id, 
                        "name": path_parts[1]
                    })
        
        logger.info(
            f"PDF listing successful | "
            f"user_id={user_id}, "
            f"total_pdfs={len(pdfs)}, "
            f"total_objects={total_objects}"
        )
        
        return JSONResponse(content={
            "user_id": user_id,
            "total_pdfs": len(pdfs), 
            "pdfs": pdfs,
            "status_code": "200"
        })
        
    except Exception as e:
        error_msg = (
            f"PDF listing failed | "
            f"user_id={user_id}, "
            f"error_type={type(e).__name__}, "
            f"error={str(e)}"
        )
        logger.error(error_msg)
        raise HTTPException(
            status_code=500, 
            detail={
                "status_code": "500",
                "error_messages": [str(e)]
            }
        )