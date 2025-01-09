from fastapi import APIRouter, HTTPException
from typing import List
from pydantic import BaseModel, Field
import openai
import json
import time
from pinecone import Pinecone
from app.service.log_client import logger
from app.service.openai_client import get_embeddings
from config import OPENAI_API_KEY, PINECONE_ANC_INDEX, PINECONE_API_KEY

router = APIRouter()
openai.api_key = OPENAI_API_KEY

# Constants
VALIDATION_DAYS = 365
DEFAULT_MAX_CHUNKS = 5
DEFAULT_SUGGESTIONS = 3
DEFAULT_MODEL = "gpt-3.5-turbo"

# Request Models
class QuestionRequest(BaseModel):
    question: str = Field(..., description="The question to be answered")
    max_chunks: int = Field(default=DEFAULT_MAX_CHUNKS, description="Maximum number of chunks to retrieve")
    model: str = Field(default=DEFAULT_MODEL, description="The OpenAI model to use")
    num_suggestions: int = Field(default=DEFAULT_SUGGESTIONS, description="Number of suggested questions to generate")

# Response Models
class Source(BaseModel):
    document_id: str
    content_type: str
    # URL-specific fields
    url: str = ""
    # Document-specific fields
    filename: str = ""
    form_url: str = ""

class QuestionResponse(BaseModel):
    question: str
    answer: str
    sources: List[Source] = []
    suggested_questions: List[str] = []
    model_used: str
    status_code: str = "200"
    found: bool
    execution_time: float

def get_system_prompt(num_suggestions: int) -> str:
    """Generate the system prompt with specific instructions for handling different content types."""
    return """You are a helpful assistant that answers questions based on provided contexts.
I will give you a question and relevant contexts. Your task is to find information in the contexts that answers the question.

Please format your response as a JSON object with these fields:
{
    "answer": "The answer found in the contexts. If any relevant information is found, include it here.",
    "found": true/false,  // true if ANY relevant information is found
    "source": {  // information about where the answer was found
        "document_id": "source document id",
        "content_type": "url or document",  // Must be either "url" or "document"
        // For URL content type:
        "url": "url if available",
        // For document content type:
        "filename": "filename if available",
        "form_url": "form url if available"  // Only for documents
    },
    "suggested_questions": []  // 3 follow-up questions if information was found
}

Important:
1. Set "found" to true if ANY relevant information exists in the contexts
2. Include partial information if that's all that's available
3. The source object MUST include different fields based on content_type:
   - For "url" content_type: include document_id, url, and content_type
   - For "document" content_type: include document_id, filename, form_url, and content_type
4. Keep your answer focused and specific to the question
5. Always preserve the original metadata structure based on content_type"""

def create_source_from_metadata(metadata: dict) -> Source:
    """Create a Source object with appropriate fields based on content type."""
    content_type = metadata.get("content_type", "url")
    
    if content_type == "url":
        return Source(
            document_id=metadata.get("document_id", ""),
            url=metadata.get("url", ""),
            content_type=content_type
        )
    else:  # document type
        return Source(
            document_id=metadata.get("document_id", ""),
            filename=metadata.get("filename", ""),
            content_type=content_type,
            form_url=metadata.get("form_url", "")
        )

@router.post("/ask-anc", response_model=QuestionResponse)
async def ask_question(request: QuestionRequest):
    start_time = time.time()
    logger.info(f"Processing question request | question={request.question}")
    
    try:
        # Get embeddings for the question
        question_embedding = get_embeddings([request.question])[0]
        
        # Initialize Pinecone
        pc = Pinecone(api_key=PINECONE_API_KEY)
        index = pc.Index(PINECONE_ANC_INDEX)
        
        # Get index stats for namespaces
        stats = index.describe_index_stats()
        namespaces = list(stats.namespaces.keys() if stats.namespaces else [])
        
        # Optimize: Query with higher top_k but fetch from fewer namespaces
        best_match = None
        best_score = -1
        
        # Only query recent namespaces (last 5)
        recent_namespaces = namespaces[-5:] if len(namespaces) > 5 else namespaces
        
        for namespace in recent_namespaces:
            try:
                results = index.query(
                    vector=question_embedding,
                    top_k=3,  # Reduced top_k
                    include_metadata=True,
                    namespace=namespace
                )
                
                # Check if we found a better match
                for match in results.matches:
                    if match.score > best_score:
                        best_score = match.score
                        best_match = match
                        
            except Exception as e:
                logger.error(f"Error querying namespace {namespace}: {str(e)}")
                continue
        
        if not best_match or best_score < 0.7:  # Add score threshold
            execution_time = round(time.time() - start_time, 2)
            return QuestionResponse(
                question=request.question,
                answer="No relevant information found.",
                sources=[],
                suggested_questions=[],
                model_used=request.model,
                status_code="200",
                found=False,
                execution_time=execution_time
            )

        # Process the best match
        metadata = best_match.metadata or {}
        text = metadata.get("text", "")
        if len(text) > 500:
            breakpoint = text.rfind(". ", 0, 500)
            if breakpoint == -1:
                breakpoint = 500
            text = text[:breakpoint + 1]
        
        # Create source based on content type
        source = create_source_from_metadata(metadata)
        
        # Prepare prompt with content type information
        system_prompt = get_system_prompt(request.num_suggestions)
        context_prompt = f"Context (from {metadata.get('document_id', '')}, content_type: {metadata.get('content_type', 'url')}):\n{text}"
        user_prompt = f"Question: {request.question}\n\nRelevant context:\n{context_prompt}"
        
        # Make OpenAI call with optimized tokens
        response = openai.ChatCompletion.create(
            model=request.model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            temperature=0.7,
            max_tokens=300  # Reduced tokens
        )
        
        try:
            gpt_response = json.loads(response.choices[0].message.content)
            execution_time = round(time.time() - start_time, 2)
            
            # Create source from GPT response
            gpt_source = gpt_response.get("source", {})
            source.content_type = gpt_source.get("content_type", source.content_type)
            
            if source.content_type == "url":
                source.url = gpt_source.get("url", source.url)
            else:
                source.filename = gpt_source.get("filename", source.filename)
                source.form_url = gpt_source.get("form_url", source.form_url)
            
            return QuestionResponse(
                question=request.question,
                answer=gpt_response.get("answer", ""),
                sources=[source],
                suggested_questions=gpt_response.get("suggested_questions", []),
                model_used=request.model,
                status_code="200",
                found=gpt_response.get("found", False),
                execution_time=execution_time
            )
            
        except json.JSONDecodeError as e:
            logger.error(f"Error parsing OpenAI response | error={str(e)}")
            execution_time = round(time.time() - start_time, 2)
            return QuestionResponse(
                question=request.question,
                answer="Error processing the response.",
                sources=[],
                suggested_questions=[],
                model_used=request.model,
                status_code="500",
                found=False,
                execution_time=execution_time
            )

    except Exception as e:
        error_msg = f"Error processing question | error={str(e)}"
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