import logging
from fastapi import APIRouter, HTTPException, Request, Form
from fastapi.responses import JSONResponse
from botocore.exceptions import ClientError
from app.service.s3_storage import (
    delete_object, 
    s3_client,
    S3_BUCKET_NAME
)
from app.service.pinecone_client import list_all_vectors, delete_vectors
from app.service.log_client import logger

router = APIRouter()

SUPPORTED_EXTENSIONS = {
    'pdf': 'application/pdf',
    'docx': 'application/vnd.openxmlformats-officedocument.wordprocessingml.document'
}

@router.delete("/{user_id}/delete_document/{document_id}")
async def delete_document(
    request: Request,
    user_id: str,
    document_id: str
):
    try:
        logger.info(f"Starting document deletion | user_id={user_id}, document_id={document_id}")
        
        # Check file extension
        file_extension = document_id.split('.')[-1].lower() if '.' in document_id else None
        if not file_extension:
            document_id = f"{document_id}.pdf"  # Default to PDF for backward compatibility
            file_extension = 'pdf'
        elif file_extension not in SUPPORTED_EXTENSIONS:
            raise HTTPException(
                status_code=400,
                detail={
                    "status_code": "400",
                    "error_messages": [f"Unsupported file type. Supported types: {', '.join(SUPPORTED_EXTENSIONS.keys())}"]
                }
            )

        object_name = f"{user_id}/{document_id}"
        logger.info(f"Checking S3 path: {object_name}")
        
        # Check S3 existence
        try:
            s3_client.head_object(Bucket=S3_BUCKET_NAME, Key=object_name)
            file_exists_in_s3 = True
        except ClientError:
            file_exists_in_s3 = False
        
        # Delete from S3 if exists
        s3_deleted = False
        if file_exists_in_s3:
            try:
                delete_object(S3_BUCKET_NAME, object_name)
                s3_deleted = True
            except Exception as e:
                logger.error(f"S3 deletion failed | error={str(e)}")
        
        # Delete vectors
        all_vectors = list_all_vectors(user_id)
        ids_to_delete = [
            v.id for v in all_vectors 
            if v.metadata.get('document_id') == document_id
        ]
        
        vectors_deleted = 0
        if ids_to_delete:
            try:
                delete_vectors(user_id, ids_to_delete)
                vectors_deleted = len(ids_to_delete)
            except Exception as e:
                logger.error(f"Vector deletion failed | error={str(e)}")

        if not s3_deleted and vectors_deleted == 0:
            raise HTTPException(
                status_code=404,
                detail={
                    "status_code": "404",
                    "error_messages": ["Document not found in storage"]
                }
            )
        
        logger.info(
            f"Document deletion completed | "
            f"user_id={user_id}, "
            f"document_id={document_id}, "
            f"s3_deleted={s3_deleted}, "
            f"vectors_deleted={vectors_deleted}"
        )
        
        return JSONResponse(
            content={
                "user_id": user_id,
                "document_id": document_id,
                "s3_deleted": s3_deleted,
                "vectors_deleted": vectors_deleted,
                "status_code": "200"
            }
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Document deletion failed | user_id={user_id}, document_id={document_id}, error={str(e)}")
        raise HTTPException(
            status_code=500,
            detail={
                "status_code": "500",
                "error_messages": [str(e)]
            }
        )