from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field
import numpy as np
from typing import List, Optional
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

class FAQCache:
    def __init__(self):
        self.questions: List[str] = []
        self.embeddings: Optional[List[List[float]]] = None
        self.answers: List[str] = []
        self.initialized: bool = False

    def is_initialized(self) -> bool:
        return self.initialized and self.embeddings is not None and len(self.questions) > 0

# Global cache instance
cache = FAQCache()

def load_faq_document() -> Document:
    """Load the FAQ document from various possible locations."""
    possible_paths = [
        FAQ_DOC_PATH,
        BASE_DIR.parent / "FAQ.docx",
        BASE_DIR.parent.parent / "FAQ.docx",
        Path("FAQ.docx"),
        # Also try with .docs extension as fallback
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
    """Initialize the FAQ cache by loading and processing the FAQ document."""
    if cache.is_initialized():
        return True

    try:
        # Load FAQ document
        doc = load_faq_document()
        faqs = []
        current_question = None
        current_answer = []

        # Parse FAQ document
        for paragraph in doc.paragraphs:
            text = paragraph.text.strip()
            if not text:
                continue

            # Assume questions end with '?'
            if text.endswith('?'):
                if current_question:
                    faqs.append({
                        'question': current_question,
                        'answer': ' '.join(current_answer)
                    })
                current_question = text
                current_answer = []
            elif current_question:
                current_answer.append(text)

        # Add last FAQ if exists
        if current_question and current_answer:
            faqs.append({
                'question': current_question,
                'answer': ' '.join(current_answer)
            })

        if not faqs:
            raise ValueError("No FAQ entries found in the document")

        # Separate questions and answers
        cache.questions = [faq['question'] for faq in faqs]
        cache.answers = [faq['answer'] for faq in faqs]

        # Get embeddings for all questions
        cache.embeddings = await openai_client.get_embeddings(cache.questions)
        cache.initialized = True

        logger.info(f"FAQ cache initialized with {len(cache.questions)} entries")
        return True

    except FileNotFoundError as e:
        logger.error(f"FAQ document not found: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail="FAQ document not found. Please ensure FAQ.docs exists in the correct location."
        )
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

async def get_answer_from_gpt(question: str, relevant_qa_pairs: List[dict]) -> tuple[str, List[str]]:
    """Generate an answer using GPT based on relevant FAQ matches."""
    try:
        context = "\n\n".join([
            f"Q: {qa['question']}\nA: {qa['answer']}"
            for qa in relevant_qa_pairs
        ])

        prompt = f"""Based on these similar FAQ entries:

{context}

Please provide a comprehensive answer to this question:
{question}

Requirements:
1. Use the provided FAQ entries as reference
2. Ensure the answer is accurate and relevant
3. Keep the response concise but complete
4. If the question cannot be fully answered from the FAQ entries, acknowledge this
5. Maintain a professional and helpful tone"""

        response = await client.chat.completions.create(
            model=MODEL,
            messages=[
                {"role": "system", "content": "You are a helpful FAQ assistant. Provide accurate answers based on the given FAQ entries."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.7,
            max_tokens=MAX_TOKENS
        )

        answer = response.choices[0].message.content.strip()
        
        # Generate suggested follow-up questions
        suggestion_prompt = f"""Based on the user's question and the provided answer, generate 3 relevant follow-up questions that a user might want to ask next. Format as a comma-separated list ONLY, with NO line breaks or numbers. Example format: "What is X?, How does Y work?, Can I do Z?"

User's question: {question}
Answer provided: {answer}

Generate exactly 3 follow-up questions in comma-separated format:"""

        suggestion_response = await client.chat.completions.create(
            model=MODEL,
            messages=[
                {"role": "system", "content": "You are a helpful FAQ assistant."},
                {"role": "user", "content": suggestion_prompt}
            ],
            temperature=0.7,
            max_tokens=200
        )
        
        # Parse and clean up suggested questions
        suggestions_text = suggestion_response.choices[0].message.content.strip()
        
        # Handle different possible formats
        if '\n' in suggestions_text:
            # Split by newline and clean up
            suggested_questions = [
                q.strip().strip('123.') for q in suggestions_text.split('\n')
                if q.strip() and not q.strip().isdigit()
            ]
        else:
            # Split by comma and clean up
            suggested_questions = [
                q.strip().strip('123.?') + '?' for q in suggestions_text.split(',')
                if q.strip()
            ]
        
        # Ensure exactly 3 questions
        suggested_questions = suggested_questions[:3]
        while len(suggested_questions) < 3:
            suggested_questions.append("What other features does Ken AI offer?")
            
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
        similarities = compute_similarity(query_embedding, cache.embeddings)

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
                answer="I apologize, but I couldn't find a closely matching question in our FAQ database. Could you please rephrase your question or provide more details?",
                confidence_score=0.0,
                suggested_questions=["How do I create an account?", "What features does Ken AI offer?", "Is there a free trial available?"],
                execution_time=round(time.time() - start_time, 2)
            )

        # Prepare relevant Q&A pairs
        relevant_pairs = [
            {
                'question': cache.questions[idx],
                'answer': cache.answers[idx],
                'similarity': sim
            }
            for idx, sim in valid_matches
        ]

        # Generate answer using GPT
        answer, suggested_questions = await get_answer_from_gpt(input_data.question, relevant_pairs)

        return FAQResponse(
            answer=answer,
            confidence_score=float(valid_matches[0][1]),  # Use highest similarity score
            suggested_questions=suggested_questions,
            execution_time=round(time.time() - start_time, 2)
        )

    except Exception as e:
        logger.error(f"Error processing FAQ request: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to process FAQ request: {str(e)}")