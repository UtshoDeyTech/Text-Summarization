from fastapi import APIRouter, HTTPException, UploadFile, File, Request
from typing import Optional
from fastapi.responses import JSONResponse
from datetime import datetime
from urllib.parse import urlparse
import re
from typing import List
from playwright.async_api import async_playwright
from langchain_text_splitters import RecursiveCharacterTextSplitter
import PyPDF2
import docx
import io
from app.service.openai_client import get_embeddings
from app.service.qdrant_client import upsert_url_vectors
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

async def extract_text_from_pdf(file_content: bytes) -> str:
    """Extract text from PDF file"""
    try:
        pdf_file = io.BytesIO(file_content)
        reader = PyPDF2.PdfReader(pdf_file)
        text = ""
        for page in reader.pages:
            text += page.extract_text() + "\n"
        return text
    except Exception as e:
        logger.error(f"PDF text extraction failed | error={str(e)}")
        raise ValueError(f"Failed to extract PDF content: {str(e)}")

async def extract_text_from_docx(file_content: bytes) -> str:
    """Extract text from DOCX file"""
    try:
        doc = docx.Document(io.BytesIO(file_content))
        text = ""
        for paragraph in doc.paragraphs:
            text += paragraph.text + "\n"
        return text
    except Exception as e:
        logger.error(f"DOCX text extraction failed | error={str(e)}")
        raise ValueError(f"Failed to extract DOCX content: {str(e)}")

@router.post("/upload_anc_global_url")
async def upload_anc_global_url(url: str):
    try:
        if not is_valid_url(url):
            raise HTTPException(
                status_code=400,
                detail={
                    "status_code": "400",
                    "error_messages": ["Invalid URL format"]
                }
            )

        logger.info(f"Processing URL | url={url}")
        chunks = await extract_text_with_playwright(url)
        embeddings = await get_embeddings(chunks)  # Added await
        
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

        namespace = upsert_url_vectors(embeddings, metadatas, ids, url)

        return JSONResponse(
            content={
                "document_id": document_id,
                "chunks_stored": len(chunks),
                "url": url,
                "namespace": namespace,
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

@router.post("/upload_anc_global_document")
async def upload_anc_global_document(
    request: Request,
    file: UploadFile = File(...),
    form_url: Optional[str] = None
):
    try:
        if not file.filename.lower().endswith(('.pdf', '.docx')):
            raise HTTPException(
                status_code=400,
                detail={
                    "status_code": "400",
                    "error_messages": ["Unsupported file format. Only PDF and DOCX files are accepted"]
                }
            )

        if form_url and not is_valid_url(form_url):
            raise HTTPException(
                status_code=400,
                detail={
                    "status_code": "400",
                    "error_messages": ["Invalid form URL format"]
                }
            )

        file_content = await file.read()
        document_id = file.filename

        if file.filename.lower().endswith('.pdf'):
            text = await extract_text_from_pdf(file_content)
        else:
            text = await extract_text_from_docx(file_content)

        chunks = text_splitter.split_text(text)
        if not chunks:
            raise ValueError("No valid text content extracted from document")

        embeddings = await get_embeddings(chunks)  # Added await
        
        ids = [f"{document_id}_{i}" for i in range(len(chunks))]
        metadatas = [
            {
                "document_id": document_id,
                "filename": file.filename,
                "content_type": "document",
                "upload_date": datetime.utcnow().isoformat(),
                "text": chunk,
                "form_url": form_url if form_url else ""
            }
            for chunk in chunks
        ]

        namespace = upsert_url_vectors(embeddings, metadatas, ids, document_id)

        return JSONResponse(
            content={
                "document_id": document_id,
                "chunks_stored": len(chunks),
                "filename": file.filename,
                "namespace": namespace,
                "form_url": form_url if form_url else "",
                "status_code": "200"
            }
        )

    except HTTPException:
        raise
    except Exception as e:
        error_msg = f"Document upload failed | document_id={document_id}, filename={file.filename}, error={str(e)}"
        logger.error(error_msg)
        raise HTTPException(
            status_code=500,
            detail={
                "status_code": "500",
                "error_messages": [str(e)]
            }
        )