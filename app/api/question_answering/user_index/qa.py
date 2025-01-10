from fastapi import APIRouter, HTTPException
from typing import List, Dict
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
from config import OPENAI_API_KEY, PINECONE_CLIENT_INDEX, PINECONE_API_KEY, MODEL, MAX_CHUNKS, NUM_SUGGESTIONS

router = APIRouter()
openai.api_key = OPENAI_API_KEY

# Constants
VALIDATION_DAYS = 365
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

def validate_days(days: int) -> int:
    """Validate and return the number of days."""
    try:
        days_int = int(days)
        return days_int if days_int > 0 else VALIDATION_DAYS
    except (TypeError, ValueError):
        return VALIDATION_DAYS

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
    "answer": "Your answer here", # The actual answer found in the context (write descriptive answer, do not provide short answer), or 'There is no relivant answer' if no answer found
    "found": true/false, # Boolean indicating if an answer was found
    "source": {{ # Source information for the specific chunk where the answer was found
        "filename": "filename here",
        "document_id": "id here",
        "file_type": "file type here",
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
9. In the summary, reference sequence numbers when connecting current answer with previous context"""

@router.post("/{user_id}/ask", response_model=QuestionResponse)
async def ask_question(user_id: str, request: QuestionRequest):
    start_time = time.time()
    
    try:
        # Get memory buffer and current sequence
        memory_entries = memory_buffer.get_memory(user_id)
        current_sequence = memory_buffer.get_current_sequence(user_id)
        
        # Quick cache check
        question_hash = get_question_hash(request.question, user_id)
        if cached_result := get_cached_response(question_hash, user_id):
            return cached_result

        # Get embeddings
        question_embedding = get_embeddings([request.question])[0]
        
        # Initialize Pinecone
        pc = Pinecone(api_key=PINECONE_API_KEY)
        client_index = pc.Index(PINECONE_CLIENT_INDEX)
        
        # Get all namespaces for the user
        client_namespaces = [ns for ns in client_index.describe_index_stats().namespaces.keys() 
                            if ns.startswith(f"{user_id}_")]
        
        if not client_namespaces:
            execution_time = round(time.time() - start_time, 2)
            return create_response(user_id, request, "", [], [], False, execution_time, current_sequence)

        # Get matches from all valid namespaces
        all_matches = []
        current_date = datetime.utcnow()
        twelve_months_ago = current_date - timedelta(days=VALIDATION_DAYS)
        
        for namespace in client_namespaces:
            # Query each namespace
            results = client_index.query(
                vector=question_embedding,
                top_k=MAX_CHUNKS,
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
            return create_response(user_id, request, "", [], [], False, execution_time, current_sequence)

        # Sort and process top matches
        sorted_matches = sorted(all_matches, key=lambda x: x.score, reverse=True)[:MAX_CHUNKS]
        
        # Prepare contexts
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

        # Prepare prompt with memory context
        system_prompt = get_system_prompt(memory_entries)
        contexts_prompt = "\n\n".join([
            f"Context {i+1} from {ctx['filename']} (ID: {ctx['document_id']}):\n{ctx['text']}"
            for i, ctx in enumerate(valid_contexts)
        ])
        
        user_prompt = f"Based on these contexts and any relevant previous conversation, answer this question: {request.question}\n\nContexts:\n{contexts_prompt}"

        # Make OpenAI call
        response = openai.ChatCompletion.create(
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