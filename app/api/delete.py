# delete.py
import logging
from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import JSONResponse
from botocore.exceptions import ClientError
from app.service.s3_storage import (
    delete_object, 
    s3_client,  # Add this import
    S3_BUCKET_NAME
)
from app.service.pinecone_client import list_all_vectors, delete_vectors
from app.service.log_client import logger

router = APIRouter()

@router.delete("/{user_id}/delete_pdf/{pdf_id}")
async def delete_pdf(
    request: Request,
    user_id: str,
    pdf_id: str
):
    try:
        logger.info(f"Starting PDF deletion process | user_id={user_id}, pdf_id={pdf_id}")
        
        # Make sure we have the .pdf extension
        if not pdf_id.endswith('.pdf'):
            pdf_id = f"{pdf_id}.pdf"

        # First check if the file exists in S3
        object_name = f"{user_id}/{pdf_id}"
        
        # For debugging, let's log the exact path we're checking
        logger.info(f"Checking S3 path: {object_name}")
        
        # Check if file exists in S3 before trying to delete
        try:
            s3_client.head_object(Bucket=S3_BUCKET_NAME, Key=object_name)
            file_exists_in_s3 = True
        except ClientError as e:
            logger.error(f"S3 check error: {str(e)}")
            file_exists_in_s3 = False
        
        # Only try to delete if file exists
        s3_deleted = False
        if file_exists_in_s3:
            try:
                logger.info(f"Deleting PDF from S3 | object_name={object_name}, bucket={S3_BUCKET_NAME}")
                delete_object(S3_BUCKET_NAME, object_name)
                s3_deleted = True
            except Exception as e:
                logger.error(f"S3 deletion failed | user_id={user_id}, pdf_id={pdf_id}, error={str(e)}")
        else:
            logger.warning(f"File not found in S3 | user_id={user_id}, pdf_id={pdf_id}")
        
        # Get and delete vectors
        logger.info(f"Fetching vectors for deletion | user_id={user_id}, pdf_id={pdf_id}")
        all_vectors = list_all_vectors(user_id)
        
        # For debugging, let's log what we find
        logger.info(f"Found {len(all_vectors)} total vectors")
        for v in all_vectors[:5]:  # Log first 5 vectors for debugging
            logger.info(f"Vector metadata: {v.metadata}")
            
        ids_to_delete = [
            v.id for v in all_vectors 
            if v.metadata.get('pdf_id') == pdf_id  # Make sure this matches how it was stored
        ]
        
        vectors_deleted = 0
        if ids_to_delete:
            logger.info(f"Deleting vectors | user_id={user_id}, vector_count={len(ids_to_delete)}")
            try:
                delete_vectors(user_id, ids_to_delete)
                vectors_deleted = len(ids_to_delete)
            except Exception as e:
                logger.error(f"Vector deletion failed | error={str(e)}")
        else:
            logger.info(f"No vectors found to delete | user_id={user_id}, pdf_id={pdf_id}")

        # If nothing was deleted from either storage, return 404
        if not s3_deleted and vectors_deleted == 0:
            logger.warning(f"PDF not found in either storage | user_id={user_id}, pdf_id={pdf_id}")
            raise HTTPException(
                status_code=404,
                detail={
                    "status_code": "404",
                    "error_messages": ["PDF not found in storage"]
                }
            )
        
        logger.info(
            f"PDF deletion completed | "
            f"user_id={user_id}, "
            f"pdf_id={pdf_id}, "
            f"s3_deleted={s3_deleted}, "
            f"vectors_deleted={vectors_deleted}"
        )
        
        return JSONResponse(
            content={
                "user_id": user_id,
                "pdf_id": pdf_id,
                "s3_deleted": s3_deleted,
                "vectors_deleted": vectors_deleted,
                "status_code": "200"
            }
        )
        
    except HTTPException:
        raise
    except Exception as e:
        error_msg = (
            f"PDF deletion failed | "
            f"user_id={user_id}, "
            f"pdf_id={pdf_id}, "
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