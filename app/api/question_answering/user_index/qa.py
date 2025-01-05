from fastapi import APIRouter, HTTPException
from typing import List
from pydantic import BaseModel, Field
import openai
import json
import time
import hashlib
from datetime import datetime, timedelta
from pinecone import Pinecone
from functools import lru_cache
from app.service.log_client import logger
from app.service.openai_client import get_embeddings
from config import OPENAI_API_KEY, PINECONE_CLIENT_INDEX, PINECONE_API_KEY

router = APIRouter()
openai.api_key = OPENAI_API_KEY

# Constants
VALIDATION_DAYS = 365
DEFAULT_MAX_CHUNKS = 5
DEFAULT_SUGGESTIONS = 3
DEFAULT_MODEL = "gpt-3.5-turbo"  # Using faster model by default
CACHE_TIMEOUT = 3600

# Request Models
class QuestionRequest(BaseModel):
    question: str = Field(..., description="The question to be answered")
    max_chunks: int = Field(default=DEFAULT_MAX_CHUNKS, description="Maximum number of chunks to retrieve")
    model: str = Field(default=DEFAULT_MODEL, description="The OpenAI model to use")
    num_suggestions: int = Field(default=DEFAULT_SUGGESTIONS, description="Number of suggested questions to generate")

# Response Models
class Source(BaseModel):
    filename: str
    document_id: str
    file_type: str
    source_type: str = "client"

class QuestionResponse(BaseModel):
    user_id: str
    question: str
    answer: str
    sources: List[Source] = []
    suggested_questions: List[str] = []
    model_used: str
    status_code: str = "200"
    found: bool
    execution_time: float

# Cache implementation
def get_question_hash(question: str, user_id: str) -> str:
    """Generate a unique hash for a question-user combination."""
    return hashlib.md5(f"{user_id}:{question}".encode()).hexdigest()

@lru_cache(maxsize=1000)
def get_cached_response(question_hash: str, user_id: str) -> QuestionResponse:
    """Get cached response if it exists."""
    return None

def cache_response(question_hash: str, user_id: str, response: QuestionResponse) -> None:
    """Cache a response for future use."""
    get_cached_response.cache_clear()  # Clear old cache entries
    get_cached_response.cache_info()   # Log cache info

def select_model(question_length: int, context_length: int) -> str:
    """Select appropriate model based on input complexity."""
    if question_length < 100 and context_length < 2000:
        return "gpt-3.5-turbo-1106"  # Faster for simple queries
    return DEFAULT_MODEL

def validate_days(days: int) -> int:
    """
    Validate and return the number of days.
    If invalid, returns default value of 365 days.
    """
    try:
        days_int = int(days)
        return days_int if days_int > 0 else VALIDATION_DAYS
    except (TypeError, ValueError):
        return VALIDATION_DAYS

def create_response(user_id: str, request: QuestionRequest, answer: str, 
                   sources: List[Source], suggested_questions: List[str], 
                   found: bool, execution_time: float) -> QuestionResponse:
    """Helper function to create consistent response format"""
    return QuestionResponse(
        user_id=user_id,
        question=request.question,
        answer=answer,
        sources=sources,
        suggested_questions=suggested_questions,
        model_used=request.model,
        status_code="200",
        found=found,
        execution_time=execution_time
    )

def get_system_prompt(num_suggestions: int) -> str:
    """Generate the system prompt with specific instructions."""
    return f"""You are a helpful assistant that answers questions based solely on the provided contexts. 
Your response must be in the following JSON format:
{{
    "answer": "Your answer here", # The actual answer found in the context (write descriptive answer, do not provide short answer), or empty string if no answer found
    "found": true/false, # Boolean indicating if an answer was found
    "source": {{ # Source information for the specific chunk where the answer was found
        "filename": "filename here",
        "document_id": "id here",
        "file_type": "file type here",
        "source_type": "client"
    }},
    "suggested_questions": [] # Array of exactly {num_suggestions} related follow-up questions if answer is found
}}

Important rules:
1. Only use information from the provided contexts
2. Only include source information for the specific chunk where you found the answer
3. If you can't find the answer, return empty string as answer, false as found, empty object as source
4. Keep the answer concise and specific
5. Format numbers, dates, and currency values appropriately
6. Generate exactly {num_suggestions} relevant follow-up questions only if an answer is found
7. Make sure suggested questions are closely related to the context and original question
8. Suggested questions should explore different aspects of the topic"""


@router.post("/{user_id}/ask", response_model=QuestionResponse)
async def ask_question(user_id: str, request: QuestionRequest):
    start_time = time.time()
    try:
        # 1. Quick cache check
        question_hash = get_question_hash(request.question, user_id)
        if cached_result := get_cached_response(question_hash, user_id):
            return cached_result

        # 2. Get embeddings only once
        question_embedding = get_embeddings([request.question])[0]
        
        # 3. Initialize Pinecone
        pc = Pinecone(api_key=PINECONE_API_KEY)
        client_index = pc.Index(PINECONE_CLIENT_INDEX)
        
        # 4. Get all namespaces for the user
        client_namespaces = [ns for ns in client_index.describe_index_stats().namespaces.keys() 
                            if ns.startswith(f"{user_id}_")]
        
        if not client_namespaces:
            execution_time = round(time.time() - start_time, 2)
            return create_response(user_id, request, "", [], [], False, execution_time)

        # 5. Get matches from all valid namespaces
        all_matches = []
        current_date = datetime.utcnow()
        twelve_months_ago = current_date - timedelta(days=VALIDATION_DAYS)
        
        for namespace in client_namespaces:
            # Query each namespace
            results = client_index.query(
                vector=question_embedding,
                top_k=request.max_chunks,
                namespace=namespace,
                include_metadata=True
            )
            
            # Process matches for this namespace
            for match in results.matches:
                if match.score < 0.6:  # Skip low relevance matches
                    continue
                    
                metadata = match.metadata or {}
                if not metadata.get("text", "").strip():
                    continue
                    
                # Check date if available
                if "upload_date" in metadata:
                    upload_date = datetime.fromisoformat(metadata["upload_date"])
                    if upload_date < twelve_months_ago:
                        continue
                
                all_matches.append(match)

        if not all_matches:
            execution_time = round(time.time() - start_time, 2)
            return create_response(user_id, request, "", [], [], False, execution_time)

        # 6. Sort and process top matches
        sorted_matches = sorted(all_matches, key=lambda x: x.score, reverse=True)[:request.max_chunks]
        
        # 7. Prepare contexts
        valid_contexts = []
        for match in sorted_matches:
            metadata = match.metadata or {}
            context = {
                "text": metadata.get("text", ""),
                "filename": metadata.get("filename", ""),
                "document_id": metadata.get("document_id", ""),
                "file_type": metadata.get("file_type", ""),
                "source_type": "client",
                "score": match.score
            }
            valid_contexts.append(context)

        # 8. Prepare prompt
        system_prompt = get_system_prompt(request.num_suggestions)
        contexts_prompt = "\n\n".join([
            f"Context {i+1} from {ctx['filename']} (ID: {ctx['document_id']}):\n{ctx['text']}"
            for i, ctx in enumerate(valid_contexts)
        ])
        
        user_prompt = f"Based on these contexts, answer this question: {request.question}\n\nContexts:\n{contexts_prompt}"

        # 9. Make OpenAI call with GPT-3.5
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo-1106",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            temperature=0.7,
            max_tokens=500
        )
        
        try:
            gpt_response = json.loads(response.choices[0].message.content)
            execution_time = round(time.time() - start_time, 2)
            
            result = create_response(
                user_id=user_id,
                request=request,
                answer=gpt_response.get("answer", ""),
                sources=[Source(**gpt_response.get("source", {}))] if gpt_response.get("found", False) else [],
                suggested_questions=gpt_response.get("suggested_questions", []),
                found=gpt_response.get("found", False),
                execution_time=execution_time
            )
            
            # Cache successful responses
            if result.found:
                cache_response(question_hash, user_id, result)
            
            return result
            
        except json.JSONDecodeError as e:
            logger.error(f"Error parsing OpenAI response | error={str(e)}")
            execution_time = round(time.time() - start_time, 2)
            return create_response(user_id, request, "", [], [], False, execution_time)

    except Exception as e:
        error_msg = f"Error processing question | user_id={user_id}, error={str(e)}"
        logger.error(error_msg)
        execution_time = round(time.time() - start_time, 2)
        raise HTTPException(
            status_code=500,
            detail={
                "status_code": "500",
                "error_messages": [str(e)],
                "execution_time": execution_time
            }
        )