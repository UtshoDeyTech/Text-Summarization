from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field
import numpy as np
from typing import List, Optional, Dict
import time
from docx import Document
from pathlib import Path
from openai import AsyncOpenAI

from app.service.log_client import logger
from app.service import openai_client
from config import OPENAI_API_KEY, MODEL

router = APIRouter()
client = AsyncOpenAI(api_key=OPENAI_API_KEY)

# Constants
TOP_K_MATCHES = 3
SIMILARITY_THRESHOLD = 0.7
MAX_TOKENS = 1000
CHUNK_SIZE = 1000  # Number of characters per chunk
CHUNK_OVERLAP = 200  # Number of characters to overlap between chunks

# Get the absolute path to the FAQ document
BASE_DIR = Path(__file__).resolve().parent
FAQ_DOC_PATH = BASE_DIR / "FAQ.docx"

class QuestionInput(BaseModel):
    question: str = Field(..., min_length=3, description="User's question")

class FAQResponse(BaseModel):
    answer: str
    confidence_score: float
    suggested_questions: List[str]
    execution_time: float

class DocumentChunk(BaseModel):
    content: str
    start_idx: int
    end_idx: int

class FAQCache:
    def __init__(self):
        self.chunks: List[DocumentChunk] = []
        self.chunk_embeddings: Optional[List[List[float]]] = None
        self.initialized: bool = False

    def is_initialized(self) -> bool:
        return self.initialized and self.chunk_embeddings is not None and len(self.chunks) > 0

# Global cache instance
cache = FAQCache()

def create_chunks(text: str) -> List[DocumentChunk]:
    """Split text into overlapping chunks."""
    chunks = []
    start = 0
    
    while start < len(text):
        # Define chunk end with overlap
        end = start + CHUNK_SIZE
        
        # If this isn't the last chunk, try to find a good breaking point
        if end < len(text):
            # Look for the last period, question mark, or exclamation mark in the overlap region
            overlap_region_start = end - CHUNK_OVERLAP
            overlap_region = text[overlap_region_start:end]
            
            # Find the last sentence boundary in the overlap region
            last_boundary = max(
                overlap_region.rfind('.'),
                overlap_region.rfind('?'),
                overlap_region.rfind('!')
            )
            
            if last_boundary != -1:
                end = overlap_region_start + last_boundary + 1
        else:
            end = len(text)
        
        chunk = DocumentChunk(
            content=text[start:end].strip(),
            start_idx=start,
            end_idx=end
        )
        chunks.append(chunk)
        
        # Move start to the beginning of the next chunk, considering overlap
        start = end - CHUNK_OVERLAP if end < len(text) else end
    
    return chunks

def load_faq_document() -> Document:
    """Load the FAQ document from various possible locations."""
    possible_paths = [
        FAQ_DOC_PATH,
        BASE_DIR.parent / "FAQ.docx",
        BASE_DIR.parent.parent / "FAQ.docx",
        Path("FAQ.docx"),
        BASE_DIR / "FAQ.docs",
        BASE_DIR.parent / "FAQ.docs",
        BASE_DIR.parent.parent / "FAQ.docs",
        Path("FAQ.docs"),
    ]
    
    for path in possible_paths:
        try:
            if path.exists():
                logger.info(f"Found FAQ document at: {path}")
                return Document(path)
        except Exception as e:
            logger.warning(f"Failed to load FAQ from {path}: {str(e)}")
            continue
    
    raise FileNotFoundError(
        f"FAQ.docs not found in any of these locations: {[str(p) for p in possible_paths]}"
    )

async def initialize_faq_cache():
    """Initialize the FAQ cache by loading, chunking, and processing the FAQ document."""
    if cache.is_initialized():
        return True

    try:
        # Load FAQ document
        doc = load_faq_document()
        
        # Combine all paragraphs into a single text
        full_text = "\n".join([paragraph.text.strip() for paragraph in doc.paragraphs if paragraph.text.strip()])
        
        # Create chunks
        cache.chunks = create_chunks(full_text)
        
        # Get embeddings for all chunks
        chunk_texts = [chunk.content for chunk in cache.chunks]
        cache.chunk_embeddings = await openai_client.get_embeddings(chunk_texts)
        cache.initialized = True

        logger.info(f"FAQ cache initialized with {len(cache.chunks)} chunks")
        return True

    except Exception as e:
        logger.error(f"Error initializing FAQ cache: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to initialize FAQ system: {str(e)}"
        )

def compute_similarity(query_embedding: List[float], doc_embeddings: List[List[float]]) -> List[float]:
    """Compute cosine similarity between query and documents."""
    query_embedding = np.array(query_embedding)
    doc_embeddings = np.array(doc_embeddings)
    
    # Normalize vectors
    query_norm = query_embedding / np.linalg.norm(query_embedding)
    doc_norms = doc_embeddings / np.linalg.norm(doc_embeddings, axis=1)[:, np.newaxis]
    
    # Compute cosine similarity
    similarities = np.dot(doc_norms, query_norm)
    return similarities.tolist()

async def get_answer_from_gpt(question: str, relevant_chunks: List[Dict]) -> tuple[str, List[str]]:
    """Generate an answer using GPT based on relevant chunks."""
    try:
        context = "\n\n".join([
            f"Content: {chunk['content']}"
            for chunk in relevant_chunks
        ])

        prompt = f"""Based on these relevant FAQ document chunks:

{context}

Please provide a comprehensive answer to this question:
{question}

Requirements:
1. Use the provided content as reference
2. Ensure the answer is accurate and relevant
3. Keep the response concise but complete
4. If the question cannot be fully answered from the provided content, acknowledge this
5. Maintain a professional and helpful tone"""

        response = await client.chat.completions.create(
            model=MODEL,
            messages=[
                {"role": "system", "content": "You are a helpful FAQ assistant. Provide accurate and concise answers based on the given content."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.7,
            max_tokens=MAX_TOKENS
        )

        answer = response.choices[0].message.content.strip()
        
        # Generate suggested follow-up questions
        suggestion_prompt = f"""Based on the user's question and the provided answer, generate 3 relevant follow-up questions. Format as a comma-separated list ONLY.

User's question: {question}
Answer provided: {answer}

Generate 3 follow-up questions:"""

        suggestion_response = await client.chat.completions.create(
            model=MODEL,
            messages=[
                {"role": "system", "content": "You are a helpful FAQ assistant."},
                {"role": "user", "content": suggestion_prompt}
            ],
            temperature=0.7,
            max_tokens=200
        )
        
        suggested_questions = [
            q.strip() + ('?' if not q.strip().endswith('?') else '')
            for q in suggestion_response.choices[0].message.content.strip().split(',')
        ][:3]
        
        # Ensure exactly 3 questions
        while len(suggested_questions) < 3:
            suggested_questions.append("What other features are available?")
            
        return answer, suggested_questions

    except Exception as e:
        logger.error(f"Error generating answer with GPT: {str(e)}")
        raise HTTPException(status_code=500, detail="Failed to generate answer")

@router.post("/faq/answer", response_model=FAQResponse)
async def get_faq_answer(input_data: QuestionInput) -> FAQResponse:
    """Get an answer to a FAQ question using vector similarity and GPT."""
    start_time = time.time()

    try:
        # Initialize cache if needed
        if not cache.is_initialized():
            await initialize_faq_cache()

        # Get embedding for the question
        query_embedding = (await openai_client.get_embeddings([input_data.question]))[0]

        # Compute similarities
        similarities = compute_similarity(query_embedding, cache.chunk_embeddings)

        # Get top K matches
        top_k_indices = np.argsort(similarities)[-TOP_K_MATCHES:][::-1]
        top_k_similarities = [similarities[i] for i in top_k_indices]
        
        # Filter matches below threshold
        valid_matches = [
            (idx, sim) for idx, sim in zip(top_k_indices, top_k_similarities)
            if sim >= SIMILARITY_THRESHOLD
        ]

        if not valid_matches:
            return FAQResponse(
                answer="I apologize, but I couldn't find relevant information in our FAQ database. Could you please rephrase your question or provide more details?",
                confidence_score=0.0,
                suggested_questions=[
                    "What features does the system offer?",
                    "How can I get started?",
                    "Where can I find more information?"
                ],
                execution_time=round(time.time() - start_time, 2)
            )

        # Prepare relevant chunks
        relevant_chunks = [
            {
                'content': cache.chunks[idx].content,
                'similarity': sim
            }
            for idx, sim in valid_matches
        ]

        # Generate answer using GPT
        answer, suggested_questions = await get_answer_from_gpt(
            input_data.question,
            relevant_chunks
        )

        return FAQResponse(
            answer=answer,
            confidence_score=float(valid_matches[0][1]),
            suggested_questions=suggested_questions,
            execution_time=round(time.time() - start_time, 2)
        )

    except Exception as e:
        logger.error(f"Error processing FAQ request: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to process FAQ request: {str(e)}")