import logging
from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import JSONResponse
from app.service.s3_storage import list_objects, S3_BUCKET_NAME
from app.service.pinecone_client import list_all_vectors, delete_vectors
from app.service.log_client import logger

router = APIRouter()

SUPPORTED_EXTENSIONS = {
    'pdf': 'application/pdf',
    'docx': 'application/vnd.openxmlformats-officedocument.wordprocessingml.document'
}

@router.post("/{user_id}/sync_pinecone")
async def sync_pinecone(request: Request, user_id: str):
    try:
        logger.info(f"Starting Pinecone sync | user_id={user_id}")
        
        response = list_objects(S3_BUCKET_NAME)
        s3_files = {
            obj['Key'].split('/')[1]
            for obj in response.get('Contents', []) 
            if any(obj['Key'].endswith(f'.{ext}') for ext in SUPPORTED_EXTENSIONS) 
            and obj['Key'].startswith(f"{user_id}/")
        }
        
        all_vectors = list_all_vectors(user_id)
        pinecone_files = {
            v.metadata['document_id']
            for v in all_vectors
        }
        
        orphaned_files = pinecone_files - s3_files
        deleted_count = 0
        
        if not orphaned_files:
            logger.info("No orphaned files found - all documents are in sync")
            return JSONResponse(content={
                "user_id": user_id,
                "sync_status": "in_sync",
                "s3_files_count": len(s3_files),
                "pinecone_files_count": len(pinecone_files),
                "common_files": list(s3_files & pinecone_files),
                "message": "All documents are synchronized between S3 and Pinecone",
                "status_code": "200"
            })
            
        for orphaned_file in orphaned_files:
            ids_to_delete = [
                v.id for v in all_vectors 
                if v.metadata['document_id'] == orphaned_file
            ]
            logger.info(f"Deleting orphaned vectors | file={orphaned_file}, count={len(ids_to_delete)}")
            delete_vectors(user_id, ids_to_delete)
            deleted_count += len(ids_to_delete)
        
        return JSONResponse(content={
            "user_id": user_id,
            "sync_status": "cleanup_performed",
            "vectors_deleted": deleted_count,
            "orphaned_files_removed": len(orphaned_files),
            "orphaned_files": list(orphaned_files),
            "status_code": "200"
        })
        
    except Exception as e:
        logger.error(f"Sync failed | user_id={user_id}, error={str(e)}")
        raise HTTPException(
            status_code=500,
            detail={"status_code": "500", "error_messages": [str(e)]}
        )