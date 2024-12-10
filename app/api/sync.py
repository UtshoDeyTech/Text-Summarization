import logging
from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import JSONResponse
from app.service.s3_storage import list_objects, delete_object, S3_BUCKET_NAME
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

async def delete_from_s3(user_id: str, files: set) -> int:
    """Delete files from S3 bucket and return count of deleted files."""
    deleted_count = 0
    for file in files:
        try:
            key = f"{user_id}/{file}"
            delete_object(S3_BUCKET_NAME, key)
            deleted_count += 1
            logger.info(f"Deleted file from S3 | user_id={user_id}, file={file}")
        except Exception as e:
            logger.error(f"Failed to delete S3 file | user_id={user_id}, file={file}, error={str(e)}")
    return deleted_count

async def delete_from_pinecone(user_id: str, files: set, index) -> int:
    """Delete files from Pinecone and return count of deleted namespaces."""
    deleted_count = 0
    for file in files:
        try:
            namespace = find_document_namespace(user_id, file)
            if namespace:
                index.delete(delete_all=True, namespace=namespace)
                deleted_count += 1
                logger.info(f"Deleted namespace from Pinecone | user_id={user_id}, file={file}, namespace={namespace}")
        except Exception as e:
            logger.error(f"Failed to delete Pinecone namespace | user_id={user_id}, file={file}, error={str(e)}")
    return deleted_count

@router.post("/{user_id}/sync_bidirectional")
async def sync_bidirectional(request: Request, user_id: str):
    try:
        logger.info(f"Starting bidirectional sync | user_id={user_id}")
        
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
        
        # Find files that exist in both systems
        common_files = s3_files & pinecone_files
        
        # Find files to delete from each system
        s3_only_files = s3_files - pinecone_files
        pinecone_only_files = pinecone_files - s3_files
        
        if not s3_only_files and not pinecone_only_files:
            logger.info("No orphaned files found - all documents are in sync")
            return JSONResponse(content={
                "user_id": user_id,
                "sync_status": "in_sync",
                "s3_files_count": len(s3_files),
                "pinecone_files_count": len(pinecone_files),
                "common_files": list(common_files),
                "message": "All documents are synchronized between S3 and Pinecone",
                "status_code": "200"
            })
        
        # Initialize Pinecone for deletion operations
        index = initialize_pinecone()
        
        # Delete orphaned files from both systems
        s3_deleted = await delete_from_s3(user_id, s3_only_files)
        pinecone_deleted = await delete_from_pinecone(user_id, pinecone_only_files, index)
        
        logger.info(
            f"Sync completed | s3_deleted={s3_deleted}, pinecone_deleted={pinecone_deleted}, "
            f"s3_orphaned={len(s3_only_files)}, pinecone_orphaned={len(pinecone_only_files)}"
        )
        
        return JSONResponse(content={
            "user_id": user_id,
            "sync_status": "cleanup_performed",
            "s3_files_deleted": {
                "count": s3_deleted,
                "files": list(s3_only_files)
            },
            "pinecone_namespaces_deleted": {
                "count": pinecone_deleted,
                "files": list(pinecone_only_files)
            },
            "remaining_files": list(common_files),
            "status_code": "200"
        })
        
    except Exception as e:
        logger.error(f"Sync failed | user_id={user_id}, error={str(e)}")
        raise HTTPException(
            status_code=500,
            detail={"status_code": "500", "error_messages": [str(e)]}
        )