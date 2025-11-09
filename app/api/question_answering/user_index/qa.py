from fastapi import APIRouter, HTTPException, Query
from typing import List, Dict, Optional
from pydantic import BaseModel, Field
from openai import AsyncOpenAI
import json
import time
import hashlib
from datetime import datetime
from app.service.qdrant_client import get_qdrant_client
from qdrant_client.models import Filter, FieldCondition, MatchValue
from functools import lru_cache
from app.service.log_client import logger
from app.service.openai_client import get_embeddings
from config import OPENAI_API_KEY, QDRANT_COLLECTION_NAME, MODEL, MAX_CHUNKS, NUM_SUGGESTIONS

router = APIRouter()
client = AsyncOpenAI(api_key=OPENAI_API_KEY)

# Constants
DEFAULT_MAX_CHUNKS = 5
DEFAULT_SUGGESTIONS = 3
DEFAULT_MODEL = "gpt-3.5-turbo"
CACHE_TIMEOUT = 3600

# Memory Buffer Models
class MemoryEntry(BaseModel):
    """Model for storing question-answer memory."""
    question: str
    answer: str
    timestamp: datetime
    sequence_number: int

class MemoryBuffer:
    """Implements a fixed-size memory buffer for each user."""
    def __init__(self, max_size: int = 3):
        self.max_size = max_size
        self.buffers: Dict[str, List[MemoryEntry]] = {}
        self.sequence_counters: Dict[str, int] = {}

    def add_entry(self, user_id: str, question: str, answer: str):
        """Add a new entry to user's memory buffer."""
        if user_id not in self.buffers:
            self.buffers[user_id] = []
            self.sequence_counters[user_id] = 0

        # Increment sequence counter
        self.sequence_counters[user_id] += 1
        
        # Add new entry
        entry = MemoryEntry(
            question=question,
            answer=answer,
            timestamp=datetime.utcnow(),
            sequence_number=self.sequence_counters[user_id]
        )
        
        # Add to buffer and maintain size
        self.buffers[user_id].append(entry)
        if len(self.buffers[user_id]) > self.max_size:
            self.buffers[user_id].pop(0)  # Remove oldest entry

    def get_memory(self, user_id: str) -> List[MemoryEntry]:
        """Get user's memory buffer."""
        return self.buffers.get(user_id, [])

    def get_current_sequence(self, user_id: str) -> int:
        """Get next sequence number for user."""
        return self.sequence_counters.get(user_id, 0) + 1

# Initialize memory buffer
memory_buffer = MemoryBuffer(max_size=3)

# Request Models
class QuestionRequest(BaseModel):
    question: str = Field(..., description="The question to be answered")

# Response Models
class Source(BaseModel):
    filename: str
    document_id: str
    file_type: str
    document_category: str 
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
    sequence_number: int = Field(default=1, description="Sequence number in conversation")

# Cache implementation
def get_question_hash(question: str, user_id: str, document_category: Optional[str] = None) -> str:
    """Generate a unique hash for a question-user-category combination."""
    cache_key = f"{user_id}:{question}"
    if document_category:
        cache_key += f":{document_category}"
    return hashlib.md5(cache_key.encode()).hexdigest()

@lru_cache(maxsize=1000)
def get_cached_response(question_hash: str, user_id: str) -> QuestionResponse:
    """Get cached response if it exists."""
    return None

def cache_response(question_hash: str, user_id: str, response: QuestionResponse) -> None:
    """Cache a response for future use."""
    get_cached_response.cache_clear()  # Clear old cache entries
    get_cached_response.cache_info()   # Log cache info

def create_response(user_id: str, request: QuestionRequest, answer: str, 
                   sources: List[Source], suggested_questions: List[str], 
                   found: bool, execution_time: float, sequence_number: int) -> QuestionResponse:
    """Helper function to create consistent response format"""
    return QuestionResponse(
        user_id=user_id,
        question=request.question,
        answer=answer,
        sources=sources,
        suggested_questions=suggested_questions,
        model_used=MODEL,
        status_code="200",
        found=found,
        execution_time=execution_time,
        sequence_number=sequence_number
    )

def get_system_prompt(memory_entries: List[MemoryEntry] = None) -> str:
    """Generate the system prompt with specific instructions and sequenced memory context."""
    memory_context = ""
    if memory_entries:
        # Create sequenced memory entries
        sequence_chunks = []
        for entry in memory_entries:
            sequence_chunks.append(
                f"Conversation #{entry.sequence_number}:\n"
                f"Question: {entry.question}\n"
                f"Answer: {entry.answer}\n"
                f"Memory Context: This was the {entry.sequence_number}{'st' if entry.sequence_number == 1 else 'nd' if entry.sequence_number == 2 else 'rd' if entry.sequence_number == 3 else 'th'} question in the sequence.\n"
            )
        memory_context = "\nPrevious conversation sequence:\n" + "\n".join(sequence_chunks)

    current_sequence = memory_entries[-1].sequence_number + 1 if memory_entries else 1

    return f"""You are a helpful assistant that answers questions based on the provided contexts. 
{memory_context}
Current Sequence: #{current_sequence} (Current question in the conversation flow)

Your response must be in the following JSON format:
{{
    "answer": "Your answer here", # The actual answer found in the context (write descriptive answer, do not provide short answer), or 'There is no relevant answer' if no answer found
    "found": true/false, # Boolean indicating if an answer was found
    "source": {{ # Source information for the specific chunk where the answer was found
        "filename": "filename here",
        "document_id": "id here",
        "file_type": "file type here",
        "document_category": "document category here",
        "source_type": "client"
    }},
    "suggested_questions": [], # Array of exactly {NUM_SUGGESTIONS} related follow-up questions if answer is found
    "sequence_number": {current_sequence}, # Current position in conversation sequence
    "summary": "Brief summary (max 50 words) connecting current answer with previous context" # Include sequence references if relevant
}}

Important rules:
1. Only use information from the provided contexts
2. Consider the sequence of previous questions when interpreting the current question
3. Only include source information for the specific chunk where you found the answer
4. If you can't find the answer, return empty string as answer, false as found, empty object as source
5. Keep the answer concise and specific
6. Format numbers, dates, and currency values appropriately
7. Generate exactly {NUM_SUGGESTIONS} relevant follow-up questions only if answer is found
8. Make sure suggested questions are closely related to the context and original question
9. In the summary, reference sequence numbers when connecting current answer with previous context
10. Include the document_category exactly as provided in the context metadata"""

@router.post("/{user_id}/ask", response_model=QuestionResponse)
async def ask_question(
    user_id: str,
    request: QuestionRequest,
    document_category: Optional[str] = Query(None, description="Filter by document category")
):
    start_time = time.time()

    try:
        # Get memory buffer and current sequence
        memory_entries = memory_buffer.get_memory(user_id)
        current_sequence = memory_buffer.get_current_sequence(user_id)

        # Log the category filter if provided
        if document_category:
            logger.info(f"Filtering search by document_category={document_category}")

        # Quick cache check (includes document_category in cache key)
        question_hash = get_question_hash(request.question, user_id, document_category)
        if cached_result := get_cached_response(question_hash, user_id):
            logger.info(f"Returning cached result for question with category={document_category}")
            return cached_result

        # Get embeddings asynchronously
        embeddings = await get_embeddings([request.question])
        question_embedding = embeddings[0]

        # Connect to Qdrant
        qdrant_client = get_qdrant_client()

        # Build search filter conditions
        filter_conditions = [
            FieldCondition(
                key="user_id",
                match=MatchValue(value=user_id)
            )
        ]

        # Add document_category filter if provided
        if document_category:
            filter_conditions.append(
                FieldCondition(
                    key="document_category",
                    match=MatchValue(value=document_category)
                )
            )
            logger.info(f"Search filter includes document_category={document_category}")

        # Search for similar vectors filtered by user_id and optionally document_category
        try:
            search_results = qdrant_client.search(
                collection_name=QDRANT_COLLECTION_NAME,
                query_vector=question_embedding,
                limit=MAX_CHUNKS * 3,  # Get more to filter by score
                score_threshold=0.6,  # Only return matches with score > 0.6
                with_payload=True,
                query_filter=Filter(must=filter_conditions)
            )
        except Exception as e:
            logger.error(f"Qdrant search error | user_id={user_id}, error={str(e)}")
            search_results = []

        if not search_results:
            execution_time = round(time.time() - start_time, 2)
            return create_response(user_id, request, "", [], [], False, execution_time, current_sequence)

        # Filter and sort matches
        all_matches = []
        for match in search_results:
            metadata = match.payload or {}
            if metadata.get("text", "").strip():
                all_matches.append(match)

        if not all_matches:
            execution_time = round(time.time() - start_time, 2)
            return create_response(user_id, request, "", [], [], False, execution_time, current_sequence)

        # Sort by score and take top MAX_CHUNKS
        sorted_matches = sorted(all_matches, key=lambda x: x.score, reverse=True)[:MAX_CHUNKS]

        # Prepare contexts with complete metadata
        valid_contexts = []
        for match in sorted_matches:
            metadata = match.payload or {}
            context = {
                "text": metadata.get("text", ""),
                "filename": metadata.get("filename", ""),
                "document_id": metadata.get("document_id", ""),
                "file_type": metadata.get("file_type", ""),
                "document_category": metadata.get("document_category", "Other"),
                "source_type": "client",
                "score": match.score
            }
            valid_contexts.append(context)

        # Prepare contexts prompt with structured metadata
        contexts_prompt = "\n\n".join([
            f"""Context {i+1}:
Document Metadata:
- Filename: {ctx['filename']}
- Document ID: {ctx['document_id']}
- File Type: {ctx['file_type']}
- Document Category: {ctx['document_category']}
- Relevance Score: {ctx['score']}

Content:
{ctx['text']}"""
            for i, ctx in enumerate(valid_contexts)
        ])
        
        # Prepare system prompt
        system_prompt = get_system_prompt(memory_entries)
        
        # Prepare user prompt with explicit instructions
        user_prompt = f"""Based on these contexts and any relevant previous conversation, answer this question: {request.question}

Important: When providing source information, use the exact document metadata (including category) from the specific context chunk where you found the answer.

Contexts:
{contexts_prompt}"""

        # Make OpenAI call with async client
        response = await client.chat.completions.create(
            model=MODEL,
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
                execution_time=execution_time,
                sequence_number=current_sequence
            )
            
            # If answer found, add to memory buffer
            if result.found:
                memory_buffer.add_entry(
                    user_id=user_id,
                    question=request.question,
                    answer=gpt_response.get("answer", "")
                )
                
                # Cache successful responses
                cache_response(question_hash, user_id, result)
            
            return result
            
        except json.JSONDecodeError as e:
            logger.error(f"Error parsing OpenAI response | error={str(e)}")
            execution_time = round(time.time() - start_time, 2)
            return create_response(user_id, request, "", [], [], False, execution_time, current_sequence)

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