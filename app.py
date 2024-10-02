import os
import uuid
import logging
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
import shutil
from pydantic import BaseModel
from pinecone_utils import initialize_pinecone, upsert_vectors, query_vectors, delete_vectors, list_all_vectors
from openai import OpenAI

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI()

# Initialize Pinecone
index = initialize_pinecone()

# Configure paths
PDF_FOLDER = "Files"
os.makedirs(PDF_FOLDER, exist_ok=True)

# Text splitter for chunking
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)

# Initialize OpenAI client
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

def pdf_to_chunks(file_path):
    try:
        with open(file_path, 'rb') as file:
            pdf = PdfReader(file)
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
        logger.error(f"Error processing PDF {file_path}: {str(e)}")
        raise

def get_embeddings(texts):
    try:
        # Ensure texts is a non-empty list of strings
        if not isinstance(texts, list) or len(texts) == 0 or not all(isinstance(t, str) for t in texts):
            raise ValueError("Input must be a non-empty list of strings")
        
        # Log the first few characters of the first text for debugging
        logger.info(f"First text (truncated): {texts[0][:100]}...")
        
        response = client.embeddings.create(
            input=texts,
            model="text-embedding-ada-002"
        )
        return [embedding.embedding for embedding in response.data]
    except Exception as e:
        logger.error(f"Error getting embeddings: {str(e)}")
        raise

@app.post("/upload_pdf")
async def upload_pdf(file: UploadFile = File(...)):
    if file.filename.split('.')[-1].lower() != 'pdf':
        raise HTTPException(status_code=400, detail="Only PDF files are allowed")
    
    pdf_id = str(uuid.uuid4())
    file_path = os.path.join(PDF_FOLDER, f"{pdf_id}.pdf")
    
    try:
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        
        logger.info(f"PDF saved to {file_path}")
        
        chunks = pdf_to_chunks(file_path)
        logger.info(f"Generated {len(chunks)} chunks from PDF")
        
        if len(chunks) == 0:
            raise ValueError("No text could be extracted from the PDF")
        
        embeddings = get_embeddings(chunks)
        logger.info(f"Generated {len(embeddings)} embeddings")
        
        ids = [f"{pdf_id}_{i}" for i in range(len(chunks))]
        metadatas = [{"pdf_id": pdf_id, "text": chunk} for chunk in chunks]
        
        upsert_vectors(index, embeddings, metadatas, ids)
        
        logger.info(f"Successfully uploaded and processed PDF: {file.filename}")
        return JSONResponse(content={"pdf_id": pdf_id, "chunks_stored": len(chunks)})
    except Exception as e:
        logger.error(f"Error uploading PDF {file.filename}: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.delete("/delete_pdf/{pdf_id}")
async def delete_pdf(pdf_id: str):
    file_path = os.path.join(PDF_FOLDER, f"{pdf_id}.pdf")
    if not os.path.exists(file_path):
        logger.warning(f"PDF file not found: {file_path}")
        raise HTTPException(status_code=404, detail="PDF not found")
    
    try:
        # Delete the PDF file
        os.remove(file_path)
        logger.info(f"Deleted PDF file: {file_path}")
        
        # Delete vectors from Pinecone
        all_vectors = list_all_vectors(index)
        logger.info(f"Total vectors in Pinecone: {len(all_vectors)}")
        
        ids_to_delete = [v.id for v in all_vectors if v.metadata.get('pdf_id') == pdf_id]
        logger.info(f"Found {len(ids_to_delete)} vectors to delete for PDF: {pdf_id}")
        
        if ids_to_delete:
            delete_vectors(index, ids_to_delete)
            logger.info(f"Deleted {len(ids_to_delete)} vectors from Pinecone for PDF: {pdf_id}")
        else:
            logger.warning(f"No vectors found in Pinecone for PDF: {pdf_id}")
        
        # Verify deletion
        remaining_vectors = [v for v in list_all_vectors(index) if v.metadata.get('pdf_id') == pdf_id]
        if remaining_vectors:
            logger.error(f"Failed to delete all vectors. {len(remaining_vectors)} vectors still remain for PDF: {pdf_id}")
            raise HTTPException(status_code=500, detail="Failed to delete all vectors from Pinecone")
        
        return JSONResponse(content={"message": f"PDF {pdf_id} and its vectors deleted successfully"})
    except Exception as e:
        logger.error(f"Error deleting PDF {pdf_id}: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/list_pdfs")
async def list_pdfs():
    try:
        pdfs = []
        for filename in os.listdir(PDF_FOLDER):
            if filename.endswith('.pdf'):
                pdf_id = filename.split('.')[0]
                pdfs.append({"id": pdf_id, "name": filename})
        logger.info(f"Listed {len(pdfs)} PDFs")
        return JSONResponse(content={"total_pdfs": len(pdfs), "pdfs": pdfs})
    except Exception as e:
        logger.error(f"Error listing PDFs: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

class SearchQuery(BaseModel):
    query: str
    n_results: int = 5

@app.post("/search_chunks")
async def search_chunks(search_query: SearchQuery):
    try:
        query_embedding = get_embeddings([search_query.query])[0]
        results = query_vectors(index, query_embedding, top_k=search_query.n_results)
        
        chunks = [
            {
                "chunk_id": result.id,
                "text": result.metadata['text'],
                "metadata": result.metadata,
                "score": result.score
            }
            for result in results
        ]
        
        logger.info(f"Found {len(chunks)} relevant chunks for query")
        return JSONResponse(content={"results": chunks})
    except Exception as e:
        logger.error(f"Error searching chunks: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/sync_pinecone")
async def sync_pinecone():
    try:
        # Get all PDF files in the folder
        pdf_files = {f.split('.')[0] for f in os.listdir(PDF_FOLDER) if f.endswith('.pdf')}

        # Get all vectors from Pinecone
        all_vectors = list_all_vectors(index)
        pinecone_pdf_ids = {v.metadata['pdf_id'] for v in all_vectors}

        # Find pdf_ids in Pinecone that don't exist in the folder
        to_delete = pinecone_pdf_ids - pdf_files

        deleted_count = 0
        for pdf_id in to_delete:
            # Delete vectors for this pdf_id
            ids_to_delete = [v.id for v in all_vectors if v.metadata['pdf_id'] == pdf_id]
            delete_vectors(index, ids_to_delete)
            deleted_count += len(ids_to_delete)

        logger.info(f"Synchronization complete. Deleted {deleted_count} vectors from {len(to_delete)} PDFs.")
        return JSONResponse(content={
            "message": f"Synchronization complete. Deleted {deleted_count} vectors from {len(to_delete)} PDFs.",
            "deleted_pdfs": list(to_delete)
        })

    except Exception as e:
        logger.error(f"An error occurred during synchronization: {str(e)}")
        raise HTTPException(status_code=500, detail=f"An error occurred during synchronization: {str(e)}")
    
@app.get("/health")
async def health_check():
    return {"status": "healthy"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=5000)