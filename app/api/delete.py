import logging
from fastapi import APIRouter, HTTPException
from fastapi.responses import JSONResponse
from app.service.s3_storage import delete_object, S3_BUCKET_NAME
from app.service.pinecone_client import initialize_pinecone, delete_vectors, list_all_vectors

router = APIRouter()
logger = logging.getLogger(__name__)
index = initialize_pinecone()

@router.delete("/delete_pdf/{pdf_id}")
async def delete_pdf(pdf_id: str):
    try:
        object_name = f"{pdf_id}.pdf"
        delete_object(S3_BUCKET_NAME, object_name)
        logger.info(f"Deleted PDF file from S3: {object_name}")
        
        all_vectors = list_all_vectors(index)
        logger.info(f"Total vectors in Pinecone: {len(all_vectors)}")
        
        ids_to_delete = [v.id for v in all_vectors if v.metadata.get('pdf_id') == pdf_id]
        logger.info(f"Found {len(ids_to_delete)} vectors to delete for PDF: {pdf_id}")
        
        if ids_to_delete:
            delete_vectors(index, ids_to_delete)
            logger.info(f"Deleted {len(ids_to_delete)} vectors from Pinecone for PDF: {pdf_id}")
        else:
            logger.warning(f"No vectors found in Pinecone for PDF: {pdf_id}")
        
        return JSONResponse(content={"message": f"PDF {pdf_id} and its vectors deleted successfully"})
    except Exception as e:
        logger.error(f"Error deleting PDF {pdf_id}: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))