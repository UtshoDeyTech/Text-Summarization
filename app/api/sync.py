import logging
from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import JSONResponse
from app.service.s3_storage import list_objects, S3_BUCKET_NAME
from app.service.pinecone_client import (
    find_document_namespace, 
    list_documents_from_pinecone,
    initialize_pinecone
)
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
        
        # Get files from S3
        response = list_objects(S3_BUCKET_NAME)
        s3_files = {
            obj['Key'].split('/')[1]
            for obj in response.get('Contents', []) 
            if any(obj['Key'].endswith(f'.{ext}') for ext in SUPPORTED_EXTENSIONS) 
            and obj['Key'].startswith(f"{user_id}/")
        }
        
        # Get files from Pinecone
        pinecone_docs = list_documents_from_pinecone(user_id)
        pinecone_files = {doc['name'] for doc in pinecone_docs}
        
        # Find orphaned files (in Pinecone but not in S3)
        orphaned_files = pinecone_files - s3_files
        
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
        
        # Initialize Pinecone once before the loop
        index = initialize_pinecone()
        
        # Delete orphaned files
        deleted_count = 0
        for orphaned_file in orphaned_files:
            try:
                namespace = find_document_namespace(user_id, orphaned_file)
                if namespace:
                    index.delete(delete_all=True, namespace=namespace)
                    deleted_count += 1
                    logger.info(f"Deleted orphaned namespace | file={orphaned_file}, namespace={namespace}")
            except Exception as e:
                logger.error(f"Failed to delete namespace for file | file={orphaned_file}, error={str(e)}")
                continue
        
        logger.info(f"Sync completed | deleted_count={deleted_count}, orphaned_files={len(orphaned_files)}")
        
        return JSONResponse(content={
            "user_id": user_id,
            "sync_status": "cleanup_performed",
            "namespaces_deleted": deleted_count,
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