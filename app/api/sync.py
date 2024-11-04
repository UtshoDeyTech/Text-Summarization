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
    try:
        logger.info(f"Starting Pinecone sync | user_id={user_id}")
        
        # List S3 objects
        logger.info(f"Fetching S3 objects | user_id={user_id}, bucket={S3_BUCKET_NAME}")
        response = list_objects(S3_BUCKET_NAME)
        pdf_files = {
            obj['Key'].split('/')[1].split('.')[0] 
            for obj in response.get('Contents', []) 
            if obj['Key'].endswith('.pdf') and obj['Key'].startswith(f"{user_id}/")
        }
        logger.info(f"S3 PDFs found | user_id={user_id}, pdf_count={len(pdf_files)}")

        # List Pinecone vectors
        logger.info(f"Fetching Pinecone vectors | user_id={user_id}")
        all_vectors = list_all_vectors(user_id)
        pinecone_pdf_ids = {v.metadata['pdf_id'] for v in all_vectors}
        logger.info(
            f"Pinecone state | "
            f"user_id={user_id}, "
            f"total_vectors={len(all_vectors)}, "
            f"unique_pdfs={len(pinecone_pdf_ids)}"
        )

        # Calculate differences
        to_delete = pinecone_pdf_ids - pdf_files
        if to_delete:
            logger.info(
                f"Found orphaned vectors | "
                f"user_id={user_id}, "
                f"pdfs_to_delete={len(to_delete)}, "
                f"pdf_ids={list(to_delete)}"
            )
        else:
            logger.info(f"No orphaned vectors found | user_id={user_id}")

        # Delete orphaned vectors
        deleted_count = 0
        for pdf_id in to_delete:
            ids_to_delete = [v.id for v in all_vectors if v.metadata['pdf_id'] == pdf_id]
            logger.info(
                f"Deleting vectors for PDF | "
                f"user_id={user_id}, "
                f"pdf_id={pdf_id}, "
                f"vector_count={len(ids_to_delete)}"
            )
            delete_vectors(user_id, ids_to_delete)
            deleted_count += len(ids_to_delete)
        
        logger.info(
            f"Sync completed successfully | "
            f"user_id={user_id}, "
            f"vectors_deleted={deleted_count}, "
            f"pdfs_deleted={len(to_delete)}"
        )
        
        return JSONResponse(content={
            "user_id": user_id,
            "vectors_deleted": deleted_count,
            "pdfs_deleted": len(to_delete),
            "deleted_pdf_ids": list(to_delete),
            "status_code": "200"
        })

    except Exception as e:
        error_msg = (
            f"Sync failed | "
            f"user_id={user_id}, "
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