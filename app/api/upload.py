import uuid
import logging
from fastapi import APIRouter, UploadFile, File, Form, HTTPException, Request
from fastapi.responses import JSONResponse
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from io import BytesIO
from app.service.s3_storage import upload_file, S3_BUCKET_NAME
from app.service.openai_client import get_embeddings
from app.service.pinecone_client import upsert_vectors
from app.service.log_client import logger
import requests
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

router = APIRouter()
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1500, chunk_overlap=200)


def create_doc_info(payload, headers):
    url = f"{os.getenv("ASP_BACKEND_URL")}/api/docinfo/createdocinfo"
    
    
    try:
        response = requests.post(
            url, 
            json=payload,
            headers=headers
        )
        
        response_data = response.json()
        print("Status Code:", response_data.get('status_code'))
        
        if response_data.get('status_code') != "200":
            error_message = response_data.get('error_messages', ['Something went wrong'])[0]
            raise Exception(error_message)
            
        print("Response Data:")
        print(response_data)
        return response_data
        
    except requests.exceptions.RequestException as e:
        print(f"Error creating doc info: {e}")
        return None

def pdf_to_chunks(file_obj):
    try:
        logger.info("Starting PDF text extraction")
        pdf = PdfReader(file_obj)
        text = ""
        for i, page in enumerate(pdf.pages):
            text += page.extract_text()
            logger.info(f"Extracted text from page | page_number={i+1}, page_text_length={len(page.extract_text())}")

        if len(text.strip()) == 0:
            logger.warning("No text extracted from PDF")
            return []

        chunks = text_splitter.split_text(text)
        logger.info(f"PDF chunking completed | total_text_length={len(text)}, chunks_created={len(chunks)}")
        return chunks
    except Exception as e:
        logger.error(f"PDF processing failed | error_type={type(e).__name__}, error={str(e)}")
        raise

@router.post("/{user_id}/upload_pdf")
async def upload_pdf(
    request: Request,
    user_id: str,
    BEARER_TOKEN: str = Form(...),
    file: UploadFile = File(...)
):
    try:
        logger.info(
            f"Starting PDF upload | "
            f"user_id={user_id}, "
            f"filename={file.filename}, "
            f"content_type={file.content_type}"
        )

        if file.filename.split('.')[-1].lower() != 'pdf':
            logger.warning(f"Invalid file type uploaded | user_id={user_id}, filename={file.filename}")
            raise HTTPException(
                status_code=400, 
                detail={
                    "status_code": "400",
                    "error_messages": ["Only PDF files are allowed"]
                }
            )
        
        pdf_id = f"{file.filename}----{str(uuid.uuid4())}"
        object_name = f"{user_id}/{pdf_id}.pdf"
        logger.info(f"Generated PDF ID | user_id={user_id}, pdf_id={pdf_id}")
        
        # Read file content
        file_content = await file.read()
        file_size = len(file_content)
        logger.info(f"Read file content | size_bytes={file_size}")
        
        # Upload to S3
        logger.info(f"Uploading to S3 | bucket={S3_BUCKET_NAME}, object_name={object_name}")
        s3_url = upload_file(BytesIO(file_content), S3_BUCKET_NAME, object_name)
        if not s3_url:
            logger.error(f"S3 upload failed | user_id={user_id}, pdf_id={pdf_id}")
            raise HTTPException(
                status_code=500, 
                detail={
                    "status_code": "500",
                    "error_messages": ["Failed to upload PDF to S3"]
                }
            )
        
        # Process PDF into chunks
        logger.info(f"Processing PDF into chunks | user_id={user_id}, pdf_id={pdf_id}")
        chunks = pdf_to_chunks(BytesIO(file_content))
        if len(chunks) == 0:
            logger.warning(f"No text extracted | user_id={user_id}, pdf_id={pdf_id}")
            raise HTTPException(
                status_code=400,
                detail={
                    "status_code": "400",
                    "error_messages": ["No text could be extracted from the PDF"]
                }
            )
        
        # Generate embeddings
        logger.info(f"Generating embeddings | chunk_count={len(chunks)}")
        embeddings = get_embeddings(chunks)
        
        # Prepare vector data
        ids = [f"{pdf_id}_{i}" for i in range(len(chunks))]
        metadatas = [{
            "pdf_id": pdf_id,
            "user_id": user_id,
            "text": chunk,
            "s3_url": s3_url,
            "filename": file.filename
        } for chunk in chunks]
        
        # Upload vectors
        logger.info(f"Upserting vectors | user_id={user_id}, vector_count={len(embeddings)}")
        upsert_vectors(user_id, embeddings, metadatas, ids)
        
        logger.info(
            f"PDF upload successful | "
            f"user_id={user_id}, "
            f"pdf_id={pdf_id}, "
            f"file_size={file_size}, "
            f"chunks_stored={len(chunks)}"
        )

        payload = {
        "userid": user_id,
        "name": pdf_id,
        "s3url": s3_url,
        "s3path": f"{S3_BUCKET_NAME}/{user_id}",
        "vectorid": f"pdf-vectors-{user_id}"
        }
        headers = {
        'accept': 'application/json',
        'Authorization': f'Bearer {BEARER_TOKEN}',
        'Content-Type': 'application/json'
        }

        response = create_doc_info(payload, headers)
        
        return JSONResponse(
            content={
                "pdf_id": pdf_id,
                "file_name": file.filename, 
                "user_id": user_id,
                "chunks_stored": len(chunks), 
                "s3_url": s3_url,
                "response": response,
                "status_code": "200"
            }
        )
    except HTTPException:
        raise
    except Exception as e:
        error_msg = (
            f"PDF upload failed | "
            f"user_id={user_id}, "
            f"filename={file.filename}, "
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