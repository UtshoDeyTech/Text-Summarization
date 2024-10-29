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
    headers = request.headers
    
    logger.info(f"Listing PDFs for user_id: {user_id}")
    
    try:
        response = list_objects(S3_BUCKET_NAME)
        pdfs = []
        
        for obj in response.get('Contents', []):
            if obj['Key'].endswith('.pdf'):
                path_parts = obj['Key'].split('/')
                if len(path_parts) > 1 and path_parts[0] == user_id:
                    pdf_id = path_parts[1].split('.')[0]
                    pdfs.append({"id": pdf_id, "name": path_parts[1]})
        
        logger.info(f"Listed {len(pdfs)} PDFs for user {user_id}")
        return JSONResponse(content={
            "user_id": user_id,
            "total_pdfs": len(pdfs), 
            "pdfs": pdfs,
            "status_code": "200"
        })
    except Exception as e:
        logger.error(f"Error listing PDFs for user {user_id}: {str(e)}")
        raise HTTPException(
            status_code=500, 
            detail={
                "status_code": "500",
                "error_messages": [str(e)]
            }
        )