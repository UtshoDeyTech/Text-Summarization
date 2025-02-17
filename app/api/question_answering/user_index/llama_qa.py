from fastapi import APIRouter, HTTPException
from typing import List, Dict
from pydantic import BaseModel, Field
from datetime import datetime
import json
import time
import hashlib
from pinecone import Pinecone
from functools import lru_cache
from app.service.log_client import logger
from app.service.openai_client import get_embeddings
from config import PINECONE_CLIENT_INDEX, PINECONE_API_KEY, MODEL, MAX_CHUNKS, NUM_SUGGESTIONS, LLAMA_URL
from langchain_ollama import OllamaLLM
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler

router = APIRouter()

# Initialize Llama
class LlamaClient:
    def __init__(self, model_name: str = "llama3:8b", timeout: int = 30):
        self.llm = OllamaLLM(
            base_url=LLAMA_URL,
            model=model_name,
            callbacks=[StreamingStdOutCallbackHandler()],
            temperature=0.7,
            timeout=timeout
        )

    async def create_chat_completion(self, messages, temperature=0.7, max_tokens=500):
        try:
            # Convert messages to a single prompt
            prompt = "\n\n".join([f"{msg['role']}: {msg['content']}" for msg in messages])
            response = self.llm.invoke(prompt)
            
            # Create a response object that mimics OpenAI's structure
            return type('Response', (), {
                'choices': [
                    type('Choice', (), {
                        'message': type('Message', (), {
                            'content': response
                        })
                    })
                ]
            })
        except Exception as e:
            raise Exception(f"Llama chat completion error: {str(e)}")

# Initialize Llama client
client = LlamaClient()

# Constants
DEFAULT_MAX_CHUNKS = 5
DEFAULT_SUGGESTIONS = 3
DEFAULT_MODEL = "llama3:8b"
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

        self.sequence_counters[user_id] += 1
        
        entry = MemoryEntry(
            question=question,
            answer=answer,
            timestamp=datetime.utcnow(),
            sequence_number=self.sequence_counters[user_id]
        )
        
        self.buffers[user_id].append(entry)
        if len(self.buffers[user_id]) > self.max_size:
            self.buffers[user_id].pop(0)

    def get_memory(self, user_id: str) -> List[MemoryEntry]:
        return self.buffers.get(user_id, [])

    def get_current_sequence(self, user_id: str) -> int:
        return self.sequence_counters.get(user_id, 0) + 1

# Initialize memory buffer
memory_buffer = MemoryBuffer(max_size=3)

# Request/Response Models remain the same
class QuestionRequest(BaseModel):
    question: str = Field(..., description="The question to be answered")

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

# Cache implementation remains the same
def get_question_hash(question: str, user_id: str) -> str:
    return hashlib.md5(f"{user_id}:{question}".encode()).hexdigest()

@lru_cache(maxsize=1000)
def get_cached_response(question_hash: str, user_id: str) -> QuestionResponse:
    return None

def cache_response(question_hash: str, user_id: str, response: QuestionResponse) -> None:
    get_cached_response.cache_clear()
    get_cached_response.cache_info()

def create_response(user_id: str, request: QuestionRequest, answer: str, 
                   sources: List[Source], suggested_questions: List[str], 
                   found: bool, execution_time: float, sequence_number: int) -> QuestionResponse:
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

# System prompt remains similar but adjusted for Llama
def get_system_prompt(memory_entries: List[MemoryEntry] = None) -> str:
    memory_context = ""
    if memory_entries:
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
Current Sequence: #{current_sequence}

Respond in JSON format:
{{
    "answer": "Your detailed answer here",
    "found": true/false,
    "source": {{
        "filename": "filename",
        "document_id": "id",
        "file_type": "type",
        "document_category": "category",
        "source_type": "client"
    }},
    "suggested_questions": [{NUM_SUGGESTIONS} questions],
    "sequence_number": {current_sequence},
    "summary": "Brief context summary"
}}"""

@router.post("/{user_id}/ask-llama", response_model=QuestionResponse)
async def ask_question(user_id: str, request: QuestionRequest):
    start_time = time.time()
    
    try:
        memory_entries = memory_buffer.get_memory(user_id)
        current_sequence = memory_buffer.get_current_sequence(user_id)
        
        question_hash = get_question_hash(request.question, user_id)
        if cached_result := get_cached_response(question_hash, user_id):
            return cached_result

        embeddings = await get_embeddings([request.question])
        question_embedding = embeddings[0] 
        
        pc = Pinecone(api_key=PINECONE_API_KEY)
        client_index = pc.Index(PINECONE_CLIENT_INDEX)
        
        client_namespaces = [ns for ns in client_index.describe_index_stats().namespaces.keys() 
                            if ns.startswith(f"{user_id}_")]
        
        if not client_namespaces:
            execution_time = round(time.time() - start_time, 2)
            return create_response(user_id, request, "", [], [], False, execution_time, current_sequence)

        all_matches = []
        for namespace in client_namespaces:
            results = client_index.query(
                vector=question_embedding,
                top_k=MAX_CHUNKS,
                namespace=namespace,
                include_metadata=True
            )
            
            for match in results.matches:
                if match.score < 0.6:
                    continue
                metadata = match.metadata or {}
                if not metadata.get("text", "").strip():
                    continue
                all_matches.append(match)

        if not all_matches:
            execution_time = round(time.time() - start_time, 2)
            return create_response(user_id, request, "", [], [], False, execution_time, current_sequence)

        sorted_matches = sorted(all_matches, key=lambda x: x.score, reverse=True)[:MAX_CHUNKS]
        
        valid_contexts = []
        for match in sorted_matches:
            metadata = match.metadata or {}
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
        
        system_prompt = get_system_prompt(memory_entries)
        user_prompt = f"""Based on these contexts and any relevant previous conversation, answer this question: {request.question}

Use the exact document metadata from the specific context chunk where you found the answer.

Contexts:
{contexts_prompt}"""

        # Use Llama client instead of OpenAI
        response = await client.create_chat_completion(
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
            
            if result.found:
                memory_buffer.add_entry(
                    user_id=user_id,
                    question=request.question,
                    answer=gpt_response.get("answer", "")
                )
                cache_response(question_hash, user_id, result)
            
            return result
            
        except json.JSONDecodeError as e:
            logger.error(f"Error parsing Llama response | error={str(e)}")
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