import logging
from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import JSONResponse
from app.service.s3_storage import delete_object, S3_BUCKET_NAME
from app.service.pinecone_client import initialize_pinecone, delete_vectors, list_all_vectors
from app.service.log_client import logger

router = APIRouter()

index = initialize_pinecone()

@router.delete("/{user_id}/delete_pdf/{pdf_id}")
async def delete_pdf(
    request: Request,
    user_id: str,
    pdf_id: str
):
    headers = request.headers
    
    logger.info(f"Processing delete for user_id: {user_id}, pdf_id: {pdf_id}")
    
    try:
        object_name = f"{user_id}/{pdf_id}.pdf"  # Include user_id in path
        delete_object(S3_BUCKET_NAME, object_name)
        logger.info(f"Deleted PDF file from S3: {object_name}")
        
        all_vectors = list_all_vectors(index)
        logger.info(f"Total vectors in Pinecone: {len(all_vectors)}")
        
        # Add user_id check in vector deletion
        ids_to_delete = [
            v.id for v in all_vectors 
            if v.metadata.get('pdf_id') == pdf_id and v.metadata.get('user_id') == user_id
        ]
        logger.info(f"Found {len(ids_to_delete)} vectors to delete for PDF: {pdf_id}")
        
        if ids_to_delete:
            delete_vectors(index, ids_to_delete)
            logger.info(f"Deleted {len(ids_to_delete)} vectors from Pinecone for PDF: {pdf_id}")
        else:
            logger.warning(f"No vectors found in Pinecone for PDF: {pdf_id}")
        
        return JSONResponse(content={
            "user_id": user_id,
            "pdf_id": pdf_id,
            "vectors_deleted": len(ids_to_delete),
            "status_code": "200"
        })
    except Exception as e:
        logger.error(f"Error deleting PDF {pdf_id} for user {user_id}: {str(e)}")
        raise HTTPException(
            status_code=500, 
            detail={
                "status_code": "500",
                "error_messages": [str(e)]
            }
        )