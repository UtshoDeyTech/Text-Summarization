import logging
from fastapi import APIRouter, HTTPException
from fastapi.responses import JSONResponse
from app.service.s3_storage import list_objects, S3_BUCKET_NAME
from app.service.pinecone_client import initialize_pinecone, delete_vectors, list_all_vectors
from app.service.log_client import logger

router = APIRouter()

index = initialize_pinecone()

@router.post("/sync_pinecone")
async def sync_pinecone():
    try:
        response = list_objects(S3_BUCKET_NAME)
        pdf_files = {obj['Key'].split('.')[0] for obj in response.get('Contents', []) if obj['Key'].endswith('.pdf')}

        all_vectors = list_all_vectors(index)
        pinecone_pdf_ids = {v.metadata['pdf_id'] for v in all_vectors}

        to_delete = pinecone_pdf_ids - pdf_files

        deleted_count = 0
        for pdf_id in to_delete:
            ids_to_delete = [v.id for v in all_vectors if v.metadata['pdf_id'] == pdf_id]
            delete_vectors(index, ids_to_delete)
            deleted_count += len(ids_to_delete)

        logger.info(f"Synchronization complete. Deleted {deleted_count} vectors from {len(to_delete)} PDFs.")
        return JSONResponse(content={
            "message": f"Synchronization complete. Deleted {deleted_count} vectors from {len(to_delete)} PDFs.",
            "deleted_pdfs": list(to_delete)
        })

    except Exception as e:
        logger.error(f"An error occurred during synchronization: {str(e)}")
        raise HTTPException(status_code=500, detail=f"An error occurred during synchronization: {str(e)}")