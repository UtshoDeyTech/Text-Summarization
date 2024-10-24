import logging
from fastapi import APIRouter, HTTPException
from fastapi.responses import JSONResponse
from app.service.s3_storage import list_objects, S3_BUCKET_NAME
from app.service.log_client import logger

router = APIRouter()

@router.get("/list_pdfs")
async def list_pdfs():
    try:
        response = list_objects(S3_BUCKET_NAME)
        pdfs = []
        for obj in response.get('Contents', []):
            if obj['Key'].endswith('.pdf'):
                pdf_id = obj['Key'].split('.')[0]
                pdfs.append({"id": pdf_id, "name": obj['Key']})
        logger.info(f"Listed {len(pdfs)} PDFs")
        return JSONResponse(content={"total_pdfs": len(pdfs), "pdfs": pdfs})
    except Exception as e:
        logger.error(f"Error listing PDFs: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))