import logging
from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from app.service.openai_client import get_embeddings
from app.service.pinecone_client import query_vectors
from app.service.log_client import logger

router = APIRouter()

class SearchQuery(BaseModel):
    query: str
    n_results: int = 5

@router.post("/{user_id}/search_chunks")
async def search_chunks(
    request: Request,
    user_id: str,
    search_query: SearchQuery
):
    headers = request.headers
    
    try:
        query_embedding = get_embeddings([search_query.query])[0]
        
        results = query_vectors(
            user_id,
            query_embedding, 
            top_k=search_query.n_results
        )
        
        chunks = [
            {
                "chunk_id": result.id,
                "text": result.metadata.get('text', ''),
                "metadata": {
                    "pdf_id": result.metadata.get('pdf_id', ''),
                    "s3_url": result.metadata.get('s3_url', ''),
                    "filename": result.metadata.get('filename', '')
                },
                "score": result.score
            }
            for result in results
        ]
        
        return JSONResponse(content={
            "user_id": user_id,
            "query": search_query.query,
            "results": chunks,
            "status_code": "200"
        })
    except Exception as e:
        logger.error(f"Error searching chunks for user {user_id}: {str(e)}")
        raise HTTPException(
            status_code=500, 
            detail={
                "status_code": "500",
                "error_messages": [str(e)]
            }
        )

@router.get("/{user_id}/search_pdf/{pdf_id}")
async def search_pdf(
    request: Request,
    user_id: str,
    pdf_id: str,
    query: str,
    n_results: int = 5
):
    headers = request.headers
    
    try:
        query_embedding = get_embeddings([query])[0]
        
        results = query_vectors(
            user_id,
            query_embedding, 
            top_k=n_results,
            filter={"pdf_id": pdf_id}
        )
        
        chunks = [
            {
                "chunk_id": result.id,
                "text": result.metadata.get('text', ''),
                "metadata": {
                    "pdf_id": result.metadata.get('pdf_id', ''),
                    "s3_url": result.metadata.get('s3_url', ''),
                    "filename": result.metadata.get('filename', '')
                },
                "score": result.score
            }
            for result in results
        ]
        
        return JSONResponse(content={
            "user_id": user_id,
            "pdf_id": pdf_id,
            "query": query,
            "results": chunks,
            "status_code": "200"
        })
    except Exception as e:
        logger.error(f"Error searching PDF {pdf_id} for user {user_id}: {str(e)}")
        raise HTTPException(
            status_code=500, 
            detail={
                "status_code": "500",
                "error_messages": [str(e)]
            }
        )