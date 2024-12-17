import logging
from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import JSONResponse
from botocore.exceptions import ClientError
from app.service.s3_storage import (
    delete_object, 
    s3_client,
    S3_BUCKET_NAME
)
from app.service.pinecone_client import delete_vectors
from app.service.log_client import logger

router = APIRouter()

SUPPORTED_EXTENSIONS = {
    'pdf': 'application/pdf',
    'docx': 'application/vnd.openxmlformats-officedocument.wordprocessingml.document',
    'csv': 'text/csv',
    'xlsx': 'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet',
    'xls': 'application/vnd.ms-excel'
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
        
        # Check S3 existence and delete if exists
        s3_deleted = False
        try:
            s3_client.head_object(Bucket=S3_BUCKET_NAME, Key=object_name)
            delete_object(S3_BUCKET_NAME, object_name)
            s3_deleted = True
            logger.info(f"Successfully deleted from S3 | object_name={object_name}")
        except ClientError as e:
            logger.warning(f"File not found in S3 or deletion failed | object_name={object_name}, error={str(e)}")
            s3_deleted = False
        
        # Delete vectors from Pinecone
        vectors_deleted = False
        try:
            vectors_deleted = delete_vectors(user_id, document_id)
            if vectors_deleted:
                logger.info(f"Successfully deleted vectors from Pinecone | document_id={document_id}")
            else:
                logger.warning(f"No vectors found to delete in Pinecone | document_id={document_id}")
        except Exception as e:
            logger.error(f"Vector deletion failed | document_id={document_id}, error={str(e)}")

        # If neither S3 nor Pinecone had the document
        if not s3_deleted and not vectors_deleted:
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