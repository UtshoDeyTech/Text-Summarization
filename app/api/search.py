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
    try:
        logger.info(f"Starting chunk search | user_id={user_id}, n_results={search_query.n_results}")
        query_embedding = get_embeddings([search_query.query])[0]
        results = query_vectors(user_id, query_embedding, top_k=search_query.n_results)
        
        chunks = [{
            "chunk_id": result.id,
            "text": result.metadata.get('text', ''),
            "metadata": {
                "document_id": result.metadata.get('document_id', ''),
                "s3_url": result.metadata.get('s3_url', ''),
                "filename": result.metadata.get('filename', ''),
                "file_type": result.metadata.get('file_type', '')
            },
            "score": result.score
        } for result in results]
        
        logger.info(f"Search successful | chunks_found={len(chunks)}")
        
        return JSONResponse(content={
            "user_id": user_id,
            "query": search_query.query,
            "results": chunks,
            "status_code": "200"
        })
    except Exception as e:
        logger.error(f"Search failed | user_id={user_id}, error={str(e)}")
        raise HTTPException(
            status_code=500,
            detail={"status_code": "500", "error_messages": [str(e)]}
        )

@router.get("/{user_id}/search_document/{document_id}")
async def search_document(
    request: Request,
    user_id: str,
    document_id: str,
    query: str,
    n_results: int = 5
):
    try:
        logger.info(f"Starting document search | user_id={user_id}, document_id={document_id}")
        query_embedding = get_embeddings([query])[0]
        results = query_vectors(
            user_id,
            query_embedding,
            top_k=n_results,
            filter={"document_id": document_id}
        )
        
        chunks = [{
            "chunk_id": result.id,
            "text": result.metadata.get('text', ''),
            "metadata": {
                "document_id": result.metadata.get('document_id', ''),
                "s3_url": result.metadata.get('s3_url', ''),
                "filename": result.metadata.get('filename', ''),
                "file_type": result.metadata.get('file_type', '')
            },
            "score": result.score
        } for result in results]
        
        logger.info(f"Document search successful | chunks_found={len(chunks)}")
        
        return JSONResponse(content={
            "user_id": user_id,
            "document_id": document_id,
            "query": query,
            "results": chunks,
            "status_code": "200"
        })
    except Exception as e:
        logger.error(f"Document search failed | user_id={user_id}, document_id={document_id}, error={str(e)}")
        raise HTTPException(
            status_code=500,
            detail={"status_code": "500", "error_messages": [str(e)]}
        )