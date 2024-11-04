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
        
        # Delete from S3
        object_name = f"{user_id}/{pdf_id}.pdf"
        logger.info(f"Deleting PDF from S3 | object_name={object_name}, bucket={S3_BUCKET_NAME}")
        delete_object(S3_BUCKET_NAME, object_name)
        
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
        else:
            logger.info(f"No vectors found to delete | user_id={user_id}, pdf_id={pdf_id}")
        
        logger.info(f"PDF deletion successful | user_id={user_id}, pdf_id={pdf_id}, vectors_deleted={len(ids_to_delete)}")
        
        return JSONResponse(content={
            "user_id": user_id,
            "pdf_id": pdf_id,
            "vectors_deleted": len(ids_to_delete),
            "status_code": "200"
        })
        
    except Exception as e:
        error_msg = f"PDF deletion failed | user_id={user_id}, pdf_id={pdf_id}, error_type={type(e).__name__}, error={str(e)}"
        logger.error(error_msg)
        raise HTTPException(
            status_code=500, 
            detail={
                "status_code": "500",
                "error_messages": [str(e)]
            }
        )