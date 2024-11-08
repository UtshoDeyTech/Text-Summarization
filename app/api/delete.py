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
    try:
        logger.info(f"Starting PDF deletion process | user_id={user_id}, pdf_id={pdf_id}")
        
        # First check if the file exists in S3
        object_name = f"{user_id}/{pdf_id}"
        try:
            # Try to delete from S3
            logger.info(f"Deleting PDF from S3 | object_name={object_name}, bucket={S3_BUCKET_NAME}")
            delete_object(S3_BUCKET_NAME, object_name)
            s3_deleted = True
        except Exception as e:
            logger.warning(f"S3 deletion failed or file not found | user_id={user_id}, pdf_id={pdf_id}, error={str(e)}")
            s3_deleted = False
        
        # Get and delete vectors
        logger.info(f"Fetching vectors for deletion | user_id={user_id}, pdf_id={pdf_id}")
        all_vectors = list_all_vectors(user_id)
        ids_to_delete = [
            v.id for v in all_vectors 
            if v.metadata.get('pdf_id') == pdf_id
        ]
        
        if ids_to_delete:
            logger.info(f"Deleting vectors | user_id={user_id}, vector_count={len(ids_to_delete)}")
            delete_vectors(user_id, ids_to_delete)
            vectors_deleted = len(ids_to_delete)
        else:
            logger.info(f"No vectors found to delete | user_id={user_id}, pdf_id={pdf_id}")
            vectors_deleted = 0

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