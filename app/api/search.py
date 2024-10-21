import logging
from fastapi import APIRouter, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from app.service.openai_client import get_embeddings
from app.service.pinecone_client import initialize_pinecone, query_vectors

router = APIRouter()
logger = logging.getLogger(__name__)
index = initialize_pinecone()

class SearchQuery(BaseModel):
    query: str
    n_results: int = 5

@router.post("/search_chunks")
async def search_chunks(search_query: SearchQuery):
    try:
        logger.info(f"Received search query: {search_query.query}")
        
        # Generate embedding for the search query
        query_embedding = get_embeddings([search_query.query])[0]
        logger.info(f"Generated embedding for the search query")
        
        # Query Pinecone index
        results = query_vectors(index, query_embedding, top_k=search_query.n_results)
        logger.info(f"Retrieved {len(results)} results from Pinecone")
        
        # Process and format the results
        chunks = [
            {
                "chunk_id": result.id,
                "text": result.metadata.get('text', ''),
                "metadata": {
                    "pdf_id": result.metadata.get('pdf_id', ''),
                    "s3_url": result.metadata.get('s3_url', '')
                },
                "score": result.score
            }
            for result in results
        ]
        
        logger.info(f"Processed {len(chunks)} relevant chunks for the query")
        
        # Return the results
        return JSONResponse(content={
            "query": search_query.query,
            "results": chunks
        })
    except Exception as e:
        logger.error(f"Error searching chunks: {str(e)}")
        raise HTTPException(status_code=500, detail=f"An error occurred while searching: {str(e)}")

@router.get("/search_pdf/{pdf_id}")
async def search_pdf(pdf_id: str, query: str, n_results: int = 5):
    try:
        logger.info(f"Received search query for PDF {pdf_id}: {query}")
        
        # Generate embedding for the search query
        query_embedding = get_embeddings([query])[0]
        logger.info(f"Generated embedding for the search query")
        
        # Query Pinecone index with PDF ID filter
        results = query_vectors(
            index, 
            query_embedding, 
            top_k=n_results,
            filter={"pdf_id": pdf_id}
        )
        logger.info(f"Retrieved {len(results)} results from Pinecone for PDF {pdf_id}")
        
        # Process and format the results
        chunks = [
            {
                "chunk_id": result.id,
                "text": result.metadata.get('text', ''),
                "metadata": {
                    "pdf_id": result.metadata.get('pdf_id', ''),
                    "s3_url": result.metadata.get('s3_url', '')
                },
                "score": result.score
            }
            for result in results
        ]
        
        logger.info(f"Processed {len(chunks)} relevant chunks for the query in PDF {pdf_id}")
        
        # Return the results
        return JSONResponse(content={
            "pdf_id": pdf_id,
            "query": query,
            "results": chunks
        })
    except Exception as e:
        logger.error(f"Error searching chunks in PDF {pdf_id}: {str(e)}")
        raise HTTPException(status_code=500, detail=f"An error occurred while searching PDF {pdf_id}: {str(e)}")