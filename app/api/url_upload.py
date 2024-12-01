import logging
from fastapi import APIRouter, HTTPException
from fastapi.responses import JSONResponse
from datetime import datetime
from langchain_community.document_loaders import UnstructuredURLLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from urllib.parse import urlparse
from app.service.openai_client import get_embeddings
from app.service.pinecone_client_url import upsert_url_vectors
from app.service.log_client import logger

router = APIRouter()

# Define text splitter here instead of importing from upload
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1500, chunk_overlap=200)

def is_valid_url(url: str) -> bool:
    """Check if the provided URL is valid"""
    try:
        result = urlparse(url)
        return all([result.scheme, result.netloc])
    except Exception:
        return False

def extract_text_from_url(url: str) -> str:
    """Extract text content from URL using UnstructuredURLLoader"""
    try:
        loader = UnstructuredURLLoader(urls=[url])
        data = loader.load()
        if not data or len(data) == 0:
            raise ValueError("No content extracted from URL")
        return data[0].page_content
    except Exception as e:
        logger.error(f"URL content extraction failed | url={url}, error={str(e)}")
        raise

@router.post("/upload_url")
async def upload_url(url: str):
    try:
        # Validate URL
        if not is_valid_url(url):
            raise HTTPException(
                status_code=400,
                detail={
                    "status_code": "400",
                    "error_messages": ["Invalid URL format"]
                }
            )

        # Extract text from URL
        logger.info(f"Processing URL | url={url}")
        text_content = extract_text_from_url(url)

        if not text_content.strip():
            raise HTTPException(
                status_code=400,
                detail={
                    "status_code": "400",
                    "error_messages": ["No text content could be extracted from URL"]
                }
            )

        # Create chunks using the local text splitter
        chunks = text_splitter.split_text(text_content)

        # Generate embeddings
        embeddings = get_embeddings(chunks)
        
        # Create metadata
        document_id = urlparse(url).netloc
        ids = [f"{document_id}_{i}" for i in range(len(chunks))]
        metadatas = [
            {
                "document_id": document_id,
                "url": url,
                "content_type": "url",
                "upload_date": datetime.utcnow().isoformat(),
                "text": chunk
            }
            for chunk in chunks
        ]

        # Upload to Pinecone
        upsert_url_vectors(embeddings, metadatas, ids, url)

        return JSONResponse(
            content={
                "document_id": document_id,
                "chunks_stored": len(chunks),
                "url": url,
                "status_code": "200"
            }
        )

    except HTTPException:
        raise
    except Exception as e:
        error_msg = f"URL upload failed | url={url}, error={str(e)}"
        logger.error(error_msg)
        raise HTTPException(
            status_code=500,
            detail={
                "status_code": "500",
                "error_messages": [str(e)]
            }
        )