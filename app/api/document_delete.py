from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import JSONResponse
from app.service.qdrant_client import delete_vectors
from app.service.log_client import logger

router = APIRouter()

@router.delete("/document_delete/{document_id}")
async def document_delete(
    request: Request,
    document_id: str
):
    try:
        logger.info(f"Starting document deletion from Qdrant | document_id={document_id}")

        # Delete vectors from Qdrant
        vectors_deleted = False
        try:
            vectors_deleted = delete_vectors(document_id)
            if vectors_deleted:
                logger.info(f"Successfully deleted vectors from Qdrant | document_id={document_id}")
            else:
                logger.warning(f"No vectors found to delete in Qdrant | document_id={document_id}")
                raise HTTPException(
                    status_code=404,
                    detail={
                        "status_code": "404",
                        "error_messages": ["Document not found in Qdrant"]
                    }
                )
        except Exception as e:
            logger.error(f"Vector deletion failed | document_id={document_id}, error={str(e)}")
            raise HTTPException(
                status_code=500,
                detail={
                    "status_code": "500",
                    "error_messages": [str(e)]
                }
            )
        
        logger.info(f"Document deletion completed | document_id={document_id}")
        
        return JSONResponse(
            content={
                "document_id": document_id,
                "vectors_deleted": vectors_deleted,
                "status_code": "200"
            }
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Document deletion failed | document_id={document_id}, error={str(e)}")
        raise HTTPException(
            status_code=500,
            detail={
                "status_code": "500",
                "error_messages": [str(e)]
            }
        )