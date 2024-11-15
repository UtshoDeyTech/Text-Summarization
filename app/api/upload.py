import uuid
import logging
from fastapi import APIRouter, UploadFile, File, Form, HTTPException, Request
from fastapi.responses import JSONResponse
from PyPDF2 import PdfReader
from docx import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from io import BytesIO
from botocore.exceptions import ClientError
from app.service.s3_storage import (
    upload_file, 
    delete_object, 
    s3_client, 
    S3_BUCKET_NAME
)
from app.service.openai_client import get_embeddings
from app.service.pinecone_client import (
    upsert_vectors,
    list_all_vectors,
    delete_vectors
)
from app.service.log_client import logger
import requests
import os
from dotenv import load_dotenv

load_dotenv()

router = APIRouter()
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1500, chunk_overlap=200)

SUPPORTED_EXTENSIONS = {
    'pdf': 'application/pdf',
    'docx': 'application/vnd.openxmlformats-officedocument.wordprocessingml.document'
}

def create_doc_info(payload, headers):
    url = "https://asp-api-dev.askken.io/api/docinfo/createdocinfo"
    try:
        response = requests.post(url, json=payload, headers=headers)
        response_data = response.json()
        
        if response_data.get('status_code') != "200":
            error_message = response_data.get('error_messages', ['Something went wrong'])[0]
            raise Exception(error_message)
            
        return response_data
    except requests.exceptions.RequestException as e:
        logger.error(f"Error creating doc info: {e}")
        return None

def extract_text_from_pdf(file_obj):
    pdf = PdfReader(file_obj)
    text = ""
    for i, page in enumerate(pdf.pages):
        text += page.extract_text()
        logger.info(f"Extracted text from PDF page | page_number={i+1}")
    return text

def extract_text_from_docx(file_obj):
    doc = Document(file_obj)
    text = ""
    for para in doc.paragraphs:
        text += para.text + "\n"
    return text

def process_document(file_obj, file_extension):
    try:
        logger.info(f"Starting text extraction | file_type={file_extension}")
        
        if file_extension == 'pdf':
            text = extract_text_from_pdf(file_obj)
        elif file_extension == 'docx':
            text = extract_text_from_docx(file_obj)
        else:
            raise ValueError(f"Unsupported file type: {file_extension}")

        if len(text.strip()) == 0:
            logger.warning("No text extracted from document")
            return []

        chunks = text_splitter.split_text(text)
        logger.info(f"Document chunking completed | total_text_length={len(text)}, chunks_created={len(chunks)}")
        return chunks
    except Exception as e:
        logger.error(f"Document processing failed | error_type={type(e).__name__}, error={str(e)}")
        raise

@router.post("/{user_id}/upload_document")
async def upload_document(
    request: Request,
    user_id: str,
    BEARER_TOKEN: str = Form(...),
    file: UploadFile = File(...)
):
    is_overwrite = True
    try:
        file_extension = file.filename.split('.')[-1].lower()
        document_id = file.filename
        object_name = f"{user_id}/{document_id}"

        if file_extension not in SUPPORTED_EXTENSIONS:
            raise HTTPException(
                status_code=400,
                detail={
                    "status_code": "400",
                    "error_messages": [f"Supported file types are: {', '.join(SUPPORTED_EXTENSIONS.keys())}"]
                }
            )

        # Check if same exact file exists
        try:
            s3_client.head_object(Bucket=S3_BUCKET_NAME, Key=object_name)
            file_exists_s3 = True
        except ClientError:
            file_exists_s3 = False

        all_vectors = list_all_vectors(user_id)
        file_exists_pinecone = any(v.metadata.get('document_id') == document_id for v in all_vectors)

        if (file_exists_s3 or file_exists_pinecone) and not is_overwrite:
            return JSONResponse(
                status_code=409,
                content={
                    "status_code": "409",
                    "error_messages": ["File already exists. Set is_overwrite=True to overwrite."],
                    "document_id": document_id,
                    "exists_in_s3": file_exists_s3,
                    "exists_in_pinecone": file_exists_pinecone
                }
            )

        # Only delete if same exact file exists and overwrite is enabled
        if (file_exists_s3 or file_exists_pinecone) and is_overwrite:
            if file_exists_s3:
                delete_object(S3_BUCKET_NAME, object_name)
            if file_exists_pinecone:
                ids_to_delete = [v.id for v in all_vectors if v.metadata.get('document_id') == document_id]
                delete_vectors(user_id, ids_to_delete)

        file_content = await file.read()
        s3_url = upload_file(BytesIO(file_content), S3_BUCKET_NAME, object_name)
        if not s3_url:
            raise HTTPException(
                status_code=500,
                detail={
                    "status_code": "500",
                    "error_messages": ["Failed to upload document to S3"]
                }
            )

        chunks = process_document(BytesIO(file_content), file_extension)
        if len(chunks) == 0:
            raise HTTPException(
                status_code=400,
                detail={
                    "status_code": "400",
                    "error_messages": ["No text could be extracted from the document"]
                }
            )

        embeddings = get_embeddings(chunks)
        ids = [f"{document_id}_{i}" for i in range(len(chunks))]
        metadatas = [{
            "document_id": document_id,
            "user_id": user_id,
            "text": chunk,
            "s3_url": s3_url,
            "filename": file.filename,
            "file_type": file_extension
        } for chunk in chunks]

        upsert_vectors(user_id, embeddings, metadatas, ids)

        payload = {
            "userid": user_id,
            "name": document_id,
            "s3url": s3_url,
            "s3path": f"{S3_BUCKET_NAME}/{user_id}",
            "vectorid": f"document-vectors-{user_id}"
        }
        headers = {
            'accept': 'application/json',
            'Authorization': f'Bearer {BEARER_TOKEN}',
            'Content-Type': 'application/json'
        }

        response = create_doc_info(payload, headers)

        return JSONResponse(
            content={
                "document_id": document_id,
                "user_id": user_id,
                "chunks_stored": len(chunks),
                "s3_url": s3_url,
                "file_type": file_extension,
                "response": response,
                "was_overwrite": is_overwrite and (file_exists_s3 or file_exists_pinecone),
                "status_code": "200"
            }
        )
    except HTTPException:
        raise
    except Exception as e:
        error_msg = f"Document upload failed | user_id={user_id}, filename={file.filename}, error={str(e)}"
        logger.error(error_msg)
        raise HTTPException(
            status_code=500,
            detail={
                "status_code": "500",
                "error_messages": [str(e)]
            }
        )