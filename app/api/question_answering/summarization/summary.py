from fastapi import APIRouter, HTTPException
from typing import List, Tuple
from pydantic import BaseModel
import time
import openai
import tiktoken
from pinecone import Pinecone
from app.service.log_client import logger
from config import OPENAI_API_KEY, PINECONE_CLIENT_INDEX, PINECONE_API_KEY, MODEL

router = APIRouter()
openai.api_key = OPENAI_API_KEY

# Constants for token management
MAX_TOKENS_PER_REQUEST = 3000
MAX_OUTPUT_TOKENS = 1000
BATCH_SIZE = 5
TOKEN_BUFFER = 100

class SummaryResponse(BaseModel):
    namespace: str
    summary: str
    total_chunks: int
    batches_processed: int
    summarization_levels: int
    execution_time: float
    status_code: str = "200"

def count_tokens(text: str) -> int:
    """Count tokens in a text string"""
    enc = tiktoken.encoding_for_model(MODEL)
    return len(enc.encode(text))

def truncate_to_token_limit(text: str, max_tokens: int) -> str:
    """Truncate text to fit within token limit"""
    enc = tiktoken.encoding_for_model(MODEL)
    tokens = enc.encode(text)
    if len(tokens) <= max_tokens:
        return text
    return enc.decode(tokens[:max_tokens])

def get_summary_prompt(level: int = 1, is_final: bool = False) -> str:
    """Get appropriate prompt based on level"""
    if is_final:
        return """Convert the policy information into a simple HTML table without any formatting characters or newlines.

Rules:
1. Use basic HTML table tags: <table>, <tr>, <th>, <td>
2. Do not add any newlines, spaces between tags, or formatting
3. Structure with these sections:
   - Policy Information
   - Coverage Details
   - Additional Information
4. Example format (but as a single line):
<table><tr><th>Field</th><th>Details</th></tr><tr><th colspan="2">Section</th></tr><tr><td>Field</td><td>Value</td></tr></table>

Create a clean, single-line HTML table without any extra characters."""
    else:
        return f"""Extract and organize policy information from Level {level} content.
Key points to extract:
1. Policy details (numbers, dates, names)
2. Coverage information and limits
3. Premium amounts and deductibles
4. Additional policy information

Format as simple text. Do not create tables yet."""

def create_safe_batch_summary(chunks: List[str], level: int) -> Tuple[str, bool]:
    """Create a summary for a batch with token limit handling"""
    combined_text = "\n\n".join([f"Chunk {i+1}:\n{chunk}" for i, chunk in enumerate(chunks)])
    total_tokens = count_tokens(combined_text)
    
    if total_tokens > MAX_TOKENS_PER_REQUEST:
        logger.warning(f"Batch too large ({total_tokens} tokens), truncating chunks")
        max_tokens_per_chunk = (MAX_TOKENS_PER_REQUEST - TOKEN_BUFFER) // len(chunks)
        truncated_chunks = [truncate_to_token_limit(chunk, max_tokens_per_chunk) for chunk in chunks]
        combined_text = "\n\n".join([f"Chunk {i+1}:\n{chunk}" for i, chunk in enumerate(truncated_chunks)])
    
    try:
        response = openai.ChatCompletion.create(
            model=MODEL,
            messages=[
                {"role": "system", "content": get_summary_prompt(level)},
                {"role": "user", "content": f"Extract and organize the key information from these policy sections:\n\n{combined_text}"}
            ],
            temperature=0.7,
            max_tokens=MAX_OUTPUT_TOKENS
        )
        return response.choices[0].message.content, True
    except Exception as e:
        logger.error(f"Error in batch summary: {str(e)}")
        return "", False

def process_chunks_hierarchically(chunks: List[str]) -> Tuple[str, int, int]:
    """Process chunks with multiple levels of summarization if needed"""
    current_chunks = chunks
    level = 1
    batch_count = 0
    
    # Process batches to collect information
    while len(current_chunks) > BATCH_SIZE:
        logger.info(f"Processing level {level} with {len(current_chunks)} chunks")
        next_level_chunks = []
        
        for i in range(0, len(current_chunks), BATCH_SIZE):
            batch = current_chunks[i:i + BATCH_SIZE]
            batch_count += 1
            
            summary, success = create_safe_batch_summary(batch, level)
            if not success:
                raise HTTPException(
                    status_code=500,
                    detail={
                        "status_code": "500",
                        "error_messages": ["Failed to process batch due to token limits"]
                    }
                )
            next_level_chunks.append(summary)
        
        current_chunks = next_level_chunks
        level += 1
        
        if level > 3:
            logger.warning("Reached maximum summarization levels")
            break
    
    # Final step: Convert to HTML table
    try:
        final_summary = openai.ChatCompletion.create(
            model=MODEL,
            messages=[
                {"role": "system", "content": get_summary_prompt(is_final=True)},
                {"role": "user", "content": "Convert this policy information into an HTML table format:\n\n" + "\n\n".join(current_chunks)}
            ],
            temperature=0.7,
            max_tokens=1000  # Strict limit for final table generation
        ).choices[0].message.content
    except Exception as e:
        logger.error(f"Error in final HTML formatting: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail={
                "status_code": "500",
                "error_messages": ["Failed to create HTML table"]
            }
        )
    
    return final_summary, batch_count, level

@router.get("/summarize/{user_id}/{vector_id}")
async def summarize(user_id: str, vector_id: str):
    start_time = time.time()
    namespace = f"{vector_id}"  # Using the full vector_id as namespace
    
    try:
        # Initialize Pinecone
        pc = Pinecone(api_key=PINECONE_API_KEY)
        index = pc.Index(PINECONE_CLIENT_INDEX)
        
        # Check if namespace exists
        stats = index.describe_index_stats()
        if namespace not in stats.namespaces:
            raise HTTPException(
                status_code=404,
                detail={
                    "status_code": "404",
                    "error_messages": [f"Policy namespace not found: {namespace}"]
                }
            )
        
        # Fetch all vectors from the namespace
        results = index.query(
            vector=[0] * 1536,
            top_k=10000,
            namespace=namespace,
            include_metadata=True
        )
        
        # Extract text from matches
        chunks = []
        for match in results.matches:
            metadata = match.metadata or {}
            if text := metadata.get("text", "").strip():
                chunks.append(text)
        
        if not chunks:
            raise HTTPException(
                status_code=404,
                detail={
                    "status_code": "404",
                    "error_messages": [f"No policy content found in namespace: {namespace}"]
                }
            )
        
        # Process chunks hierarchically
        logger.info(f"Processing {len(chunks)} chunks from policy namespace {namespace}")
        final_summary, batches_processed, levels = process_chunks_hierarchically(chunks)
        execution_time = round(time.time() - start_time, 2)
        
        return SummaryResponse(
            namespace=namespace,
            summary=final_summary,
            total_chunks=len(chunks),
            batches_processed=batches_processed,
            summarization_levels=levels,
            execution_time=execution_time
        )
        
    except Exception as e:
        error_msg = f"Error generating policy summary | namespace={namespace}, error={str(e)}"
        logger.error(error_msg)
        raise HTTPException(
            status_code=500,
            detail={
                "status_code": "500",
                "error_messages": [str(e)],
                "execution_time": round(time.time() - start_time, 2)
            }
        )