import logging
from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import JSONResponse
from app.service.s3_storage import delete_object, S3_BUCKET_NAME
from app.service.pinecone_client import list_all_vectors, delete_vectors
from app.service.log_client import logger

router = APIRouter()

@router.delete("/{user_id}/delete_pdf/{pdf_id}")
async def delete_pdf(
    request: Request,
    user_id: str,
    pdf_id: str
):
    headers = request.headers
    
    try:
        object_name = f"{user_id}/{pdf_id}.pdf"
        delete_object(S3_BUCKET_NAME, object_name)
        
        all_vectors = list_all_vectors(user_id)
        ids_to_delete = [
            v.id for v in all_vectors 
            if v.metadata.get('pdf_id') == pdf_id
        ]
        
        if ids_to_delete:
            delete_vectors(user_id, ids_to_delete)
        
        return JSONResponse(content={
            "user_id": user_id,
            "pdf_id": pdf_id,
            "vectors_deleted": len(ids_to_delete),
            "status_code": "200"
        })
    except Exception as e:
        logger.error(f"Error deleting PDF {pdf_id} for user {user_id}: {str(e)}")
        raise HTTPException(
            status_code=500, 
            detail={
                "status_code": "500",
                "error_messages": [str(e)]
            }
        )