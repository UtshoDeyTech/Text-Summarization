from fastapi import APIRouter, HTTPException
from fastapi.responses import JSONResponse
from datetime import datetime
from urllib.parse import urlparse
from bs4 import BeautifulSoup
import re
from typing import List
from playwright.async_api import async_playwright
from langchain.text_splitter import RecursiveCharacterTextSplitter
from app.service.openai_client import get_embeddings
from app.service.pinecone_client_url import upsert_url_vectors
from app.service.log_client import logger

router = APIRouter()

# Define text splitter
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1500, chunk_overlap=200)

def is_valid_url(url: str) -> bool:
    """Check if the provided URL is valid"""
    try:
        result = urlparse(url)
        return all([result.scheme, result.netloc])
    except Exception:
        return False

def clean_text(text: str) -> str:
    """Clean extracted text using regex patterns"""
    # Remove extra whitespace and newlines
    text = re.sub(r'\s+', ' ', text)
    # Remove special characters but keep basic punctuation
    text = re.sub(r'[^\w\s.,!?-]', '', text)
    # Remove any URLs
    text = re.sub(r'http\S+|www.\S+', '', text)
    return text.strip()

async def extract_text_with_playwright(url: str) -> List[str]:
    """Extract text from dynamic websites using Playwright"""
    try:
        async with async_playwright() as p:
            # Launch browser in headless mode
            browser = await p.chromium.launch(headless=True)
            context = await browser.new_context(
                user_agent='Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
            )
            
            # Create new page and navigate to URL
            page = await context.new_page()
            await page.goto(url, wait_until='networkidle', timeout=30000)
            
            # Wait for content to load
            await page.wait_for_timeout(2000)  # Additional wait for dynamic content
            
            # Extract text from specific elements
            content_elements = await page.evaluate("""
                () => {
                    const elements = document.querySelectorAll('p, h1, h2, h3, h4, h5, h6, article, section, div');
                    const textContents = [];
                    elements.forEach(element => {
                        const text = element.innerText;
                        if (text && text.length > 20) {
                            textContents.push(text);
                        }
                    });
                    return textContents;
                }
            """)
            
            # Clean and process extracted text
            cleaned_texts = [clean_text(text) for text in content_elements if clean_text(text)]
            full_text = '\n'.join(cleaned_texts)
            
            # Close browser
            await browser.close()
            
            if not full_text.strip():
                raise ValueError("No text content could be extracted from URL")
                
            # Split text into chunks
            chunks = text_splitter.split_text(full_text)
            
            if not chunks:
                raise ValueError("No valid chunks created from text content")
                
            logger.info(f"Successfully scraped webpage | chunks={len(chunks)}")
            return chunks
            
    except Exception as e:
        logger.error(f"Error scraping webpage with Playwright | url={url} | error={str(e)}")
        raise ValueError(f"Failed to extract content: {str(e)}")

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
        chunks = await extract_text_with_playwright(url)

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