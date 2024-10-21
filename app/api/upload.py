import uuid
import logging
from fastapi import APIRouter, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from io import BytesIO
from app.service.s3_storage import upload_file, S3_BUCKET_NAME
from app.service.openai_client import get_embeddings
from app.service.pinecone_client import initialize_pinecone, upsert_vectors

router = APIRouter()
logger = logging.getLogger(__name__)
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
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

@router.post("/upload_pdf")
async def upload_pdf(file: UploadFile = File(...)):
    if file.filename.split('.')[-1].lower() != 'pdf':
        raise HTTPException(status_code=400, detail="Only PDF files are allowed")
    
    pdf_id = str(uuid.uuid4())
    object_name = f"{pdf_id}.pdf"
    
    try:
        file_content = await file.read()
        
        s3_url = upload_file(BytesIO(file_content), S3_BUCKET_NAME, object_name)
        if not s3_url:
            raise HTTPException(status_code=500, detail="Failed to upload PDF to S3")
        
        logger.info(f"PDF uploaded to S3: {s3_url}")
        
        pdf_file = BytesIO(file_content)
        chunks = pdf_to_chunks(pdf_file)
        logger.info(f"Generated {len(chunks)} chunks from PDF")
        
        if len(chunks) == 0:
            raise ValueError("No text could be extracted from the PDF")
        
        embeddings = get_embeddings(chunks)
        logger.info(f"Generated {len(embeddings)} embeddings")
        
        ids = [f"{pdf_id}_{i}" for i in range(len(chunks))]
        metadatas = [{"pdf_id": pdf_id, "text": chunk, "s3_url": s3_url} for chunk in chunks]
        
        upsert_vectors(index, embeddings, metadatas, ids)
        
        logger.info(f"Successfully uploaded and processed PDF: {file.filename}")
        return JSONResponse(content={"pdf_id": pdf_id, "chunks_stored": len(chunks), "s3_url": s3_url})
    except Exception as e:
        logger.error(f"Error uploading PDF {file.filename}: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))