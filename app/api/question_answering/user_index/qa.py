from fastapi import APIRouter, HTTPException
from fastapi.responses import JSONResponse
from typing import List, Optional
from pydantic import BaseModel, Field
import openai
import json
import time
from datetime import datetime, timedelta
from pinecone import Pinecone
from app.service.log_client import logger
from app.service.openai_client import get_embeddings
from config import OPENAI_API_KEY, PINECONE_CLIENT_INDEX, PINECONE_API_KEY

router = APIRouter()
openai.api_key = OPENAI_API_KEY

VALIDATION_DAYS = 365  # 12 months
DEFAULT_MAX_CHUNKS = 5
DEFAULT_SUGGESTIONS = 3

def validate_days(days: int) -> int:
    """
    Validate and return the number of days.
    If invalid, returns default value of 365 days.
    
    Args:
        days (int): Number of days to validate
        
    Returns:
        int: Validated number of days
    """
    try:
        days_int = int(days)
        return days_int if days_int > 0 else VALIDATION_DAYS
    except (TypeError, ValueError):
        return VALIDATION_DAYS

# Request Models
class QuestionRequest(BaseModel):
    question: str = Field(..., description="The question to be answered")
    max_chunks: int = Field(default=DEFAULT_MAX_CHUNKS, description="Maximum number of chunks to retrieve")
    model: str = Field(default="gpt-4", description="The OpenAI model to use")
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

@router.post("/{user_id}/ask", response_model=QuestionResponse)
async def ask_question(user_id: str, request: QuestionRequest):
    start_time = time.time()
    try:
        question_embedding = get_embeddings([request.question])[0]
        current_date = datetime.utcnow()
        validation_days = validate_days(VALIDATION_DAYS)
        twelve_months_ago = current_date - timedelta(days=validation_days)
        
        pc = Pinecone(api_key=PINECONE_API_KEY)
        client_index = pc.Index(PINECONE_CLIENT_INDEX)
        
        client_namespaces = [ns for ns in client_index.describe_index_stats().namespaces.keys() 
                            if ns.startswith(f"{user_id}_")]
        
        if not client_namespaces:
            execution_time = round(time.time() - start_time, 2)
            return create_response(user_id, request, "", [], [], False, execution_time)

        # Filter valid namespaces
        valid_namespaces = []
        for namespace in client_namespaces:
            results = client_index.query(
                vector=question_embedding,
                top_k=1,
                namespace=namespace,
                include_metadata=True
            )
            
            if not results.matches:
                continue
                
            metadata = results.matches[0].metadata or {}
            if "upload_date" not in metadata:
                continue
                
            upload_date = datetime.fromisoformat(metadata["upload_date"])
            if upload_date >= twelve_months_ago:
                valid_namespaces.append(namespace)

        if not valid_namespaces:
            execution_time = round(time.time() - start_time, 2)
            return create_response(user_id, request, "", [], [], False, execution_time)

        # Query valid namespaces
        client_contexts = []
        for namespace in valid_namespaces:
            results = client_index.query(
                vector=question_embedding,
                top_k=request.max_chunks,
                namespace=namespace,
                include_metadata=True
            )
            
            for match in results.matches:
                metadata = match.metadata or {}
                context = {
                    "text": metadata.get("text", ""),
                    "filename": metadata.get("filename", ""),
                    "document_id": metadata.get("document_id", ""),
                    "file_type": metadata.get("file_type", ""),
                    "source_type": "client",
                    "score": match.score
                }
                if context["text"].strip():
                    client_contexts.append(context)

        if not client_contexts:
            execution_time = round(time.time() - start_time, 2)
            return create_response(user_id, request, "", [], [], False, execution_time)

        system_prompt = f"""You are a helpful assistant that answers questions based solely on the provided contexts. 
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
    "suggested_questions": [] # Array of exactly {request.num_suggestions} related follow-up questions if answer is found
}}

Important rules:
1. Only use information from the provided contexts
2. Only include source information for the specific chunk where you found the answer
3. If you can't find the answer, return empty string as answer, false as found, empty object as source
4. Keep the answer concise and specific
5. Format numbers, dates, and currency values appropriately
6. Generate exactly {request.num_suggestions} relevant follow-up questions only if an answer is found
7. Make sure suggested questions are closely related to the context and original question
8. Suggested questions should explore different aspects of the topic"""

        contexts_prompt = "\n\n".join([
            f"Context {i+1} from {ctx['filename']} (ID: {ctx['document_id']}):\n{ctx['text']}"
            for i, ctx in enumerate(client_contexts)
        ])

        user_prompt = f"""Based on these contexts, answer this question: {request.question}

Contexts:
{contexts_prompt}"""

        response = openai.ChatCompletion.create(
            model=request.model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            temperature=0.7
        )
        
        try:
            gpt_response = json.loads(response.choices[0].message.content)
            execution_time = round(time.time() - start_time, 2)
            
            return create_response(
                user_id=user_id,
                request=request,
                answer=gpt_response.get("answer", ""),
                sources=[Source(**gpt_response.get("source", {}))] if gpt_response.get("found", False) else [],
                suggested_questions=gpt_response.get("suggested_questions", []),
                found=gpt_response.get("found", False),
                execution_time=execution_time
            )
            
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