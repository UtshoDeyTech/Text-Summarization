import logging
from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import JSONResponse
from app.service.s3_storage import list_objects, S3_BUCKET_NAME
from app.service.pinecone_client import list_all_vectors, delete_vectors
from app.service.log_client import logger

router = APIRouter()

@router.post("/{user_id}/sync_pinecone")
async def sync_pinecone(
    request: Request,
    user_id: str
):
    headers = request.headers
    
    try:
        response = list_objects(S3_BUCKET_NAME)
        pdf_files = {
            obj['Key'].split('/')[1].split('.')[0] 
            for obj in response.get('Contents', []) 
            if obj['Key'].endswith('.pdf') and obj['Key'].startswith(f"{user_id}/")
        }

        all_vectors = list_all_vectors(user_id)
        pinecone_pdf_ids = {v.metadata['pdf_id'] for v in all_vectors}
        to_delete = pinecone_pdf_ids - pdf_files

        deleted_count = 0
        for pdf_id in to_delete:
            ids_to_delete = [v.id for v in all_vectors if v.metadata['pdf_id'] == pdf_id]
            delete_vectors(user_id, ids_to_delete)
            deleted_count += len(ids_to_delete)
        
        return JSONResponse(content={
            "user_id": user_id,
            "vectors_deleted": deleted_count,
            "pdfs_deleted": len(to_delete),
            "deleted_pdf_ids": list(to_delete),
            "status_code": "200"
        })

    except Exception as e:
        logger.error(f"Error during synchronization for user {user_id}: {str(e)}")
        raise HTTPException(
            status_code=500, 
            detail={
                "status_code": "500",
                "error_messages": [str(e)]
            }
        )