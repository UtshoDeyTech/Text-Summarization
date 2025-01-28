from fastapi import APIRouter, HTTPException, UploadFile, File
from typing import List, Tuple, Optional
from pydantic import BaseModel, Field
import time
import tiktoken
from openai import AsyncOpenAI
import PyPDF2
import io
from app.service.log_client import logger
from config import OPENAI_API_KEY, MODEL

router = APIRouter()
client = AsyncOpenAI(api_key=OPENAI_API_KEY)

# Constants for token management
MAX_TOKENS_PER_REQUEST = 3000
MAX_OUTPUT_TOKENS = 1000
BATCH_SIZE = 5
TOKEN_BUFFER = 100
MAX_PAGES = 10

# Pydantic Models
class TokenUsage(BaseModel):
    """Model for tracking token usage in API requests"""
    prompt_tokens: int = Field(
        description="Number of tokens used in the prompt",
        ge=0
    )
    completion_tokens: int = Field(
        description="Number of tokens used in the completion",
        ge=0
    )
    total_tokens: int = Field(
        description="Total number of tokens used",
        ge=0
    )

class SummaryResponse(BaseModel):
    """Model for PDF summary response"""
    summary: str = Field(
        description="Generated HTML table summary of the PDF content"
    )
    total_pages: int = Field(
        description="Total number of pages in the original PDF",
        ge=0
    )
    pages_processed: int = Field(
        description="Number of pages that were actually processed",
        ge=0
    )
    batches_processed: int = Field(
        description="Number of batches processed during summarization",
        ge=0
    )
    summarization_levels: int = Field(
        description="Number of hierarchical summarization levels used",
        ge=1,
        le=3
    )
    execution_time: float = Field(
        description="Total execution time in seconds",
        ge=0
    )
    token_usage: TokenUsage = Field(
        description="Token usage statistics for the API calls"
    )
    status_code: str = Field(
        default="200",
        description="HTTP status code of the response"
    )

class ErrorDetail(BaseModel):
    """Model for error response details"""
    status_code: str = Field(
        description="HTTP status code of the error"
    )
    error_messages: List[str] = Field(
        description="List of error messages"
    )
    execution_time: Optional[float] = Field(
        None,
        description="Total execution time before error occurred",
        ge=0
    )

class BatchSummaryResult(BaseModel):
    """Model for batch summary results"""
    content: str = Field(
        description="Summarized content from the batch"
    )
    success: bool = Field(
        description="Whether the batch processing was successful"
    )
    token_usage: TokenUsage = Field(
        description="Token usage for this batch"
    )

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

Create a clean, single-line raw HTML table (do not add any css code) without any extra characters."""
    else:
        return f"""Extract and organize policy information from Level {level} content.
Key points to extract:
1. Policy details (numbers, dates, names)
2. Coverage information and limits
3. Premium amounts and deductibles
4. Additional policy information

Format as simple text. Do not create tables yet."""

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

async def create_safe_batch_summary(chunks: List[str], level: int) -> BatchSummaryResult:
    """Create a summary for a batch with token limit handling"""
    
    # Calculate maximum tokens per chunk
    max_tokens_per_chunk = (MAX_TOKENS_PER_REQUEST - TOKEN_BUFFER) // len(chunks)
    truncated_chunks = [truncate_to_token_limit(chunk, max_tokens_per_chunk) for chunk in chunks]
    
    # Join chunks with clear separation
    combined_text = "\n\n".join([f"Chunk {i+1}:\n{chunk}" for i, chunk in enumerate(truncated_chunks)])
    
    try:
        response = await client.chat.completions.create(
            model=MODEL,
            messages=[
                {"role": "system", "content": get_summary_prompt(level)},
                {"role": "user", "content": f"Extract and organize the key information from these policy sections:\n\n{combined_text}"}
            ],
            temperature=0.7,
            max_tokens=MAX_OUTPUT_TOKENS
        )

        token_usage = TokenUsage(
            prompt_tokens=response.usage.prompt_tokens,
            completion_tokens=response.usage.completion_tokens,
            total_tokens=response.usage.total_tokens
        )
        
        return BatchSummaryResult(
            content=response.choices[0].message.content,
            success=True,
            token_usage=token_usage
        )
    except Exception as e:
        logger.error(f"Error in batch summary: {str(e)}")
        return BatchSummaryResult(
            content="",
            success=False,
            token_usage=TokenUsage(prompt_tokens=0, completion_tokens=0, total_tokens=0)
        )

async def process_chunks_hierarchically(chunks: List[str]) -> Tuple[str, int, int, TokenUsage]:
    """Process chunks with multiple levels of summarization if needed"""
    current_chunks = chunks
    level = 1
    batch_count = 0
    total_token_usage = TokenUsage(prompt_tokens=0, completion_tokens=0, total_tokens=0)
    
    # Process batches to collect information
    while len(current_chunks) > BATCH_SIZE:
        logger.info(f"Processing level {level} with {len(current_chunks)} chunks")
        next_level_chunks = []
        
        for i in range(0, len(current_chunks), BATCH_SIZE):
            batch = current_chunks[i:i + BATCH_SIZE]
            batch_count += 1
            
            result = await create_safe_batch_summary(batch, level)
            if not result.success:
                raise HTTPException(
                    status_code=500,
                    detail=ErrorDetail(
                        status_code="500",
                        error_messages=["Failed to process batch"]
                    ).dict()
                )

            # Accumulate token usage
            total_token_usage.prompt_tokens += result.token_usage.prompt_tokens
            total_token_usage.completion_tokens += result.token_usage.completion_tokens
            total_token_usage.total_tokens += result.token_usage.total_tokens
            
            next_level_chunks.append(result.content)
        
        current_chunks = next_level_chunks
        level += 1
        
        if level > 3:
            logger.warning("Reached maximum summarization levels")
            break
    
    # Final step: Convert to HTML table
    try:
        final_response = await client.chat.completions.create(
            model=MODEL,
            messages=[
                {"role": "system", "content": get_summary_prompt(is_final=True)},
                {"role": "user", "content": "Convert this policy information into an HTML table format:\n\n" + "\n\n".join(current_chunks)}
            ],
            temperature=0.7,
            max_tokens=1000  # Strict limit for final table generation
        )

        # Add final step token usage
        total_token_usage.prompt_tokens += final_response.usage.prompt_tokens
        total_token_usage.completion_tokens += final_response.usage.completion_tokens
        total_token_usage.total_tokens += final_response.usage.total_tokens

        final_summary = final_response.choices[0].message.content
    except Exception as e:
        logger.error(f"Error in final HTML formatting: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=ErrorDetail(
                status_code="500",
                error_messages=["Failed to create HTML table"]
            ).dict()
        )
    
    return final_summary, batch_count, level, total_token_usage

def extract_text_from_pdf(pdf_file: bytes) -> Tuple[List[str], int]:
    """Extract text from PDF file and return chunks of text along with total pages"""
    try:
        # Create PDF reader object
        pdf_reader = PyPDF2.PdfReader(io.BytesIO(pdf_file))
        total_pages = len(pdf_reader.pages)
        
        # Limit to first 10 pages
        pages_to_process = min(total_pages, MAX_PAGES)
        
        # Extract text from pages
        chunks = []
        for page_num in range(pages_to_process):
            page = pdf_reader.pages[page_num]
            text = page.extract_text()
            if text.strip():  # Only add non-empty pages
                chunks.append(text)
        
        return chunks, total_pages
    except Exception as e:
        logger.error(f"Error processing PDF: {str(e)}")
        raise HTTPException(
            status_code=400,
            detail=ErrorDetail(
                status_code="400",
                error_messages=["Failed to process PDF file"]
            ).dict()
        )

@router.post("/summarize/pdf", response_model=SummaryResponse)
async def summarize_pdf(file: UploadFile = File(...)) -> SummaryResponse:
    """Endpoint to process and summarize PDF content"""
    start_time = time.time()
    
    if not file.filename.lower().endswith('.pdf'):
        raise HTTPException(
            status_code=400,
            detail=ErrorDetail(
                status_code="400",
                error_messages=["Only PDF files are supported"]
            ).dict()
        )
    
    try:
        # Read PDF file
        pdf_content = await file.read()
        chunks, total_pages = extract_text_from_pdf(pdf_content)
        
        if not chunks:
            raise HTTPException(
                status_code=400,
                detail=ErrorDetail(
                    status_code="400",
                    error_messages=["No readable text found in PDF"]
                ).dict()
            )
        
        # Process chunks hierarchically
        logger.info(f"Processing {len(chunks)} chunks from PDF")
        final_summary, batches_processed, levels, token_usage = await process_chunks_hierarchically(chunks)
        execution_time = round(time.time() - start_time, 2)
        
        return SummaryResponse(
            summary=final_summary,
            total_pages=total_pages,
            pages_processed=min(total_pages, MAX_PAGES),
            batches_processed=batches_processed,
            summarization_levels=levels,
            execution_time=execution_time,
            token_usage=token_usage
        )
        
    except Exception as e:
        error_msg = f"Error generating PDF summary | error={str(e)}"
        logger.error(error_msg)
        raise HTTPException(
            status_code=500,
            detail=ErrorDetail(
                status_code="500",
                error_messages=[str(e)],
                execution_time=round(time.time() - start_time, 2)
            ).dict()
        )