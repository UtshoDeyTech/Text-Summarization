import uuid
import logging
from fastapi import APIRouter, UploadFile, File, HTTPException, Header, Path, Request
from fastapi.responses import JSONResponse
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from io import BytesIO
from app.service.s3_storage import upload_file, S3_BUCKET_NAME
from app.service.openai_client import get_embeddings
from app.service.pinecone_client import initialize_pinecone, upsert_vectors
from app.service.log_client import logger

router = APIRouter()

text_splitter = RecursiveCharacterTextSplitter(chunk_size=1500, chunk_overlap=200)
index = initialize_pinecone()

def pdf_to_chunks(file_obj):
    try:
        pdf = PdfReader(file_obj)
        logger.info(f"PDF has {len(pdf.pages)} pages")
        text = ""
        for i, page in enumerate(pdf.pages):
            page_text = page.extract_text()
            logger.info(f"Page {i+1} has {len(page_text)} characters")
            text += page_text

        logger.info(f"Total extracted text has {len(text)} characters")

        if len(text.strip()) == 0:
            logger.warning("Extracted text is empty")
            return []

        chunks = text_splitter.split_text(text)
        logger.info(f"Split text into {len(chunks)} chunks")

        return chunks
    except Exception as e:
        logger.error(f"Error processing PDF: {str(e)}")
        raise

@router.post("/{user_id}/upload_pdf")
async def upload_pdf(
    request: Request,
    user_id: str,
    file: UploadFile = File(...)
):
    
    headers = request.headers
  
    logger.info(f"Processing upload for user_id: {user_id}")
    
    if file.filename.split('.')[-1].lower() != 'pdf':
        raise HTTPException(
            status_code=400, 
            detail={
                "status_code": "400",
                "error_messages": ["Only PDF files are allowed"]
            }
        )
    
    pdf_id = str(uuid.uuid4())
    object_name = f"{user_id}/{pdf_id}.pdf"  # Include user_id in S3 path
    
    try:
        file_content = await file.read()
        
        s3_url = upload_file(BytesIO(file_content), S3_BUCKET_NAME, object_name)
        if not s3_url:
            raise HTTPException(
                status_code=500, 
                detail={
                    "status_code": "500",
                    "error_messages": ["Failed to upload PDF to S3"]
                }
            )
        
        logger.info(f"PDF uploaded to S3: {s3_url}")
        
        pdf_file = BytesIO(file_content)
        chunks = pdf_to_chunks(pdf_file)
        logger.info(f"Generated {len(chunks)} chunks from PDF")
        
        if len(chunks) == 0:
            raise HTTPException(
                status_code=400,
                detail={
                    "status_code": "400",
                    "error_messages": ["No text could be extracted from the PDF"]
                }
            )
        
        embeddings = get_embeddings(chunks)
        logger.info(f"Generated {len(embeddings)} embeddings")
        
        ids = [f"{pdf_id}_{i}" for i in range(len(chunks))]
        metadatas = [{
            "pdf_id": pdf_id,
            "user_id": user_id,  # Include user_id in metadata
            "text": chunk,
            "s3_url": s3_url,
            "filename": file.filename
        } for chunk in chunks]
        
        upsert_vectors(index, embeddings, metadatas, ids)
        
        logger.info(f"Successfully uploaded and processed PDF: {file.filename} for user: {user_id}")
        return JSONResponse(
            content={
                "pdf_id": pdf_id, 
                "user_id": user_id,
                "chunks_stored": len(chunks), 
                "s3_url": s3_url,
                "status_code": "200"
            }
        )
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error uploading PDF {file.filename} for user {user_id}: {str(e)}")
        raise HTTPException(
            status_code=500, 
            detail={
                "status_code": "500",
                "error_messages": [str(e)]
            }
        )