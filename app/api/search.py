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
        logger.info(
            f"Starting chunk search | "
            f"user_id={user_id}, "
            f"query_length={len(search_query.query)}, "
            f"n_results={search_query.n_results}"
        )
        
        logger.info(f"Generating query embedding | user_id={user_id}")
        query_embedding = get_embeddings([search_query.query])[0]
        
        logger.info(f"Querying vectors | user_id={user_id}, top_k={search_query.n_results}")
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
        
        logger.info(
            f"Chunk search successful | "
            f"user_id={user_id}, "
            f"chunks_found={len(chunks)}, "
            f"min_score={min([c['score'] for c in chunks], default=0):.3f}, "
            f"max_score={max([c['score'] for c in chunks], default=0):.3f}"
        )
        
        return JSONResponse(content={
            "user_id": user_id,
            "query": search_query.query,
            "results": chunks,
            "status_code": "200"
        })
    except Exception as e:
        error_msg = (
            f"Chunk search failed | "
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

@router.get("/{user_id}/search_pdf/{pdf_id}")
async def search_pdf(
    request: Request,
    user_id: str,
    pdf_id: str,
    query: str,
    n_results: int = 5
):
    try:
        logger.info(
            f"Starting PDF search | "
            f"user_id={user_id}, "
            f"pdf_id={pdf_id}, "
            f"query_length={len(query)}, "
            f"n_results={n_results}"
        )
        
        logger.info(f"Generating query embedding | user_id={user_id}, pdf_id={pdf_id}")
        query_embedding = get_embeddings([query])[0]
        
        logger.info(
            f"Querying vectors with filter | "
            f"user_id={user_id}, "
            f"pdf_id={pdf_id}, "
            f"top_k={n_results}"
        )
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
        
        logger.info(
            f"PDF search successful | "
            f"user_id={user_id}, "
            f"pdf_id={pdf_id}, "
            f"chunks_found={len(chunks)}, "
            f"min_score={min([c['score'] for c in chunks], default=0):.3f}, "
            f"max_score={max([c['score'] for c in chunks], default=0):.3f}"
        )
        
        return JSONResponse(content={
            "user_id": user_id,
            "pdf_id": pdf_id,
            "query": query,
            "results": chunks,
            "status_code": "200"
        })
    except Exception as e:
        error_msg = (
            f"PDF search failed | "
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