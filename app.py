import os
import uuid
import logging
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse
import chromadb
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
import shutil
from pydantic import BaseModel

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI()

# Initialize ChromaDB client
try:
    client = chromadb.HttpClient(host="localhost", port=8000)
    collection = client.get_or_create_collection(name="pdf_vectors")
    logger.info("Successfully connected to ChromaDB")
except Exception as e:
    logger.error(f"Failed to connect to ChromaDB: {str(e)}")
    raise

# Configure paths
PDF_FOLDER = "Files"
os.makedirs(PDF_FOLDER, exist_ok=True)

# Text splitter for chunking
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)

def pdf_to_chunks(file_path):
    try:
        with open(file_path, 'rb') as file:
            pdf = PdfReader(file)
            text = ""
            for page in pdf.pages:
                text += page.extract_text()
        return text_splitter.split_text(text)
    except Exception as e:
        logger.error(f"Error processing PDF {file_path}: {str(e)}")
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
        
        chunks = pdf_to_chunks(file_path)
        collection.add(
            documents=chunks,
            metadatas=[{"pdf_id": pdf_id} for _ in chunks],
            ids=[f"{pdf_id}_{i}" for i in range(len(chunks))]
        )
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
        
        # Delete chunks from ChromaDB
        results = collection.get(where={"pdf_id": pdf_id})
        if not results['ids']:
            logger.warning(f"No chunks found in ChromaDB for PDF: {pdf_id}")
        else:
            chunk_count = len(results['ids'])
            collection.delete(ids=results['ids'])
            logger.info(f"Deleted {chunk_count} chunks from ChromaDB for PDF: {pdf_id}")
        
        # Verify deletion
        if os.path.exists(file_path):
            logger.error(f"Failed to delete PDF file: {file_path}")
            raise HTTPException(status_code=500, detail="Failed to delete PDF file")
        
        verify_results = collection.get(where={"pdf_id": pdf_id})
        if verify_results['ids']:
            logger.error(f"Failed to delete all chunks from ChromaDB for PDF: {pdf_id}")
            raise HTTPException(status_code=500, detail="Failed to delete all chunks from ChromaDB")
        
        return JSONResponse(content={"message": f"PDF {pdf_id} and its chunks deleted successfully"})
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

@app.get("/pdf_chunks/{pdf_id}")
async def get_pdf_chunks(pdf_id: str):
    try:
        results = collection.get(where={"pdf_id": pdf_id})
        return JSONResponse(content={"pdf_id": pdf_id, "total_chunks": len(results['ids'])})
    except Exception as e:
        logger.error(f"Error getting chunks for PDF {pdf_id}: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

class SearchQuery(BaseModel):
    query: str
    n_results: int = 5

@app.post("/search_chunks")
async def search_chunks(search_query: SearchQuery):
    try:
        results = collection.query(
            query_texts=[search_query.query],
            n_results=search_query.n_results,
            include=['documents', 'metadatas', 'distances']
        )
        
        if not results['ids'][0]:
            raise HTTPException(status_code=404, detail="No matching chunks found")
        
        chunks = [
            {
                "chunk_id": chunk_id,
                "text": document,
                "metadata": metadata,
                "distance": distance
            }
            for chunk_id, document, metadata, distance in zip(
                results['ids'][0], results['documents'][0], results['metadatas'][0], results['distances'][0]
            )
        ]
        
        logger.info(f"Found {len(chunks)} relevant chunks for query")
        return JSONResponse(content={"results": chunks})
    except Exception as e:
        logger.error(f"Error searching chunks: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/sync_chromadb")
async def sync_chromadb():
    try:
        # Get all PDF files in the folder
        pdf_files = {f.split('.')[0] for f in os.listdir(PDF_FOLDER) if f.endswith('.pdf')}

        # Get all unique pdf_ids in ChromaDB
        all_metadatas = collection.get()['metadatas']
        chroma_pdf_ids = {metadata['pdf_id'] for metadata in all_metadatas if metadata}

        # Find pdf_ids in ChromaDB that don't exist in the folder
        to_delete = chroma_pdf_ids - pdf_files

        deleted_count = 0
        for pdf_id in to_delete:
            # Get all chunks for this pdf_id
            results = collection.get(where={"pdf_id": pdf_id})
            if results['ids']:
                # Delete these chunks from ChromaDB
                collection.delete(ids=results['ids'])
                deleted_count += len(results['ids'])

        logger.info(f"Synchronization complete. Deleted {deleted_count} chunks from {len(to_delete)} PDFs.")
        return JSONResponse(content={
            "message": f"Synchronization complete. Deleted {deleted_count} chunks from {len(to_delete)} PDFs.",
            "deleted_pdfs": list(to_delete)
        })

    except Exception as e:
        logger.error(f"An error occurred during synchronization: {str(e)}")
        raise HTTPException(status_code=500, detail=f"An error occurred during synchronization: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=5000)