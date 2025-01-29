from fastapi import APIRouter, HTTPException
from typing import List, Tuple, Optional
from pydantic import BaseModel, Field
import time
import tiktoken
from openai import AsyncOpenAI
import PyPDF2
import io
from app.service.log_client import logger
from config import OPENAI_API_KEY, MODEL, S3_BUCKET_NAME
from app.service import s3_storage
from pydantic import validator

router = APIRouter()
client = AsyncOpenAI(api_key=OPENAI_API_KEY)

# Constants for token management
MAX_TOKENS_PER_REQUEST = 3000
MAX_OUTPUT_TOKENS = 1000
BATCH_SIZE = 5
TOKEN_BUFFER = 100
MAX_PAGES = 10

class S3URLInput(BaseModel):
    """Model for S3 URL input"""
    url: str = Field(
        description="S3 URL of the PDF file to process"
    )
    
    @validator('url')
    def validate_s3_url(cls, v):
        # Check if it's a valid S3 URL format
        if not v.startswith(f"https://{S3_BUCKET_NAME}.s3."):
            raise ValueError("Invalid S3 URL format")
        
        # Check if it ends with .pdf (case insensitive)
        if not v.lower().endswith('.pdf'):
            raise ValueError("URL must point to a PDF file")
            
        return v

class TokenUsage(BaseModel):
    """Model for tracking token usage in API requests"""
    prompt_tokens: int = Field(description="Number of tokens used in the prompt", ge=0)
    completion_tokens: int = Field(description="Number of tokens used in the completion", ge=0)
    total_tokens: int = Field(description="Total number of tokens used", ge=0)

class SummaryResponse(BaseModel):
    """Model for PDF summary response"""
    summary: str = Field(description="Generated HTML table summary of the PDF content")
    total_pages: int = Field(description="Total number of pages in the original PDF", ge=0)
    pages_processed: int = Field(description="Number of pages that were actually processed", ge=0)
    batches_processed: int = Field(description="Number of batches processed during summarization", ge=0)
    summarization_levels: int = Field(description="Number of hierarchical summarization levels used", ge=1, le=3)
    execution_time: float = Field(description="Total execution time in seconds", ge=0)
    token_usage: TokenUsage = Field(description="Token usage statistics for the API calls")
    status_code: str = Field(default="200", description="HTTP status code of the response")

class ErrorDetail(BaseModel):
    """Model for error response details"""
    status_code: str = Field(description="HTTP status code of the error")
    error_messages: List[str] = Field(description="List of error messages")
    execution_time: Optional[float] = Field(None, description="Total execution time before error occurred", ge=0)

class BatchSummaryResult(BaseModel):
    """Model for batch summary results"""
    content: str = Field(description="Summarized content from the batch")
    success: bool = Field(description="Whether the batch processing was successful")
    token_usage: TokenUsage = Field(description="Token usage for this batch")

# Helper functions remain the same
def get_summary_prompt(level: int = 1, is_final: bool = False) -> str:
    """Get appropriate prompt based on level"""
    if is_final:
        return """Convert the policy information into a structured HTML table following these strict rules:

1. Table Structure:
   - Start with Policy Information section (policyholder details, policy numbers, dates)
   - Follow with Coverage Details section (coverage types, limits, terms)
   - End with Additional Information section (general terms, conditions)
   
2. Format as a clean single-line HTML table:
<table><tr><th>Field</th><th>Details</th></tr><tr><th colspan="2">Policy Information</th></tr><tr><td>Field</td><td>Value</td></tr></table>

Important rules:
- Only include verified information with actual values
- Use "Not specified" for unknown values
- Remove empty cells or rows
- No duplicate information
- No placeholder text
- No chunk sections or numbering
- No line breaks (<br>) within cells
- No CSS or styling"""
    else:
        return f"""Extract key policy information from Level {level} content. Focus on:
1. Core policy details (numbers, dates, policyholder info)
2. Coverage information
3. Important terms and conditions

Format as simple text with clear labels. Keep only verified information."""

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
    max_tokens_per_chunk = (MAX_TOKENS_PER_REQUEST - TOKEN_BUFFER) // len(chunks)
    truncated_chunks = [truncate_to_token_limit(chunk, max_tokens_per_chunk) for chunk in chunks]
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

            total_token_usage.prompt_tokens += result.token_usage.prompt_tokens
            total_token_usage.completion_tokens += result.token_usage.completion_tokens
            total_token_usage.total_tokens += result.token_usage.total_tokens
            
            next_level_chunks.append(result.content)
        
        current_chunks = next_level_chunks
        level += 1
        
        if level > 3:
            logger.warning("Reached maximum summarization levels")
            break
    
    try:
        final_response = await client.chat.completions.create(
            model=MODEL,
            messages=[
                {"role": "system", "content": get_summary_prompt(is_final=True)},
                {"role": "user", "content": "Convert this policy information into an HTML table format:\n\n" + "\n\n".join(current_chunks)}
            ],
            temperature=0.7,
            max_tokens=1000
        )

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

async def get_pdf_from_s3(url: str) -> bytes:
    """Fetch PDF from S3 using the provided URL"""
    try:
        # Parse the URL to get the object key
        # Example URL: https://python-api-ai-dev.s3.us-east-1.amazonaws.com/169/Policy/2Y236487- Policy Travlers - Panchal Investments Inc..pdf
        from urllib.parse import unquote
        
        # Remove the base S3 URL and get the path
        base_url = f"https://{S3_BUCKET_NAME}.s3.us-east-1.amazonaws.com/"
        if base_url not in url:
            raise ValueError("Invalid S3 URL format")
            
        # Get the object key and decode URL encoding
        object_key = unquote(url.replace(base_url, ""))
        logger.info(f"Extracted object key: {object_key}")
        
        # Verify bucket exists and is accessible
        if not s3_storage.verify_bucket(S3_BUCKET_NAME):
            raise HTTPException(
                status_code=404,
                detail=ErrorDetail(
                    status_code="404",
                    error_messages=["S3 bucket not accessible"]
                ).dict()
            )
        
        # Get object from S3
        response = s3_storage.s3_client.get_object(Bucket=S3_BUCKET_NAME, Key=object_key)
        pdf_content = response['Body'].read()
        return pdf_content
        
    except ValueError as e:
        logger.error(f"URL format error: {str(e)}")
        raise HTTPException(
            status_code=400,
            detail=ErrorDetail(
                status_code="400",
                error_messages=[str(e)]
            ).dict()
        )
    except s3_storage.ClientError as e:
        logger.error(f"Error getting object from S3: {str(e)}")
        raise HTTPException(
            status_code=404,
            detail=ErrorDetail(
                status_code="404",
                error_messages=["PDF file not found in S3"]
            ).dict()
        )
    except Exception as e:
        logger.error(f"Unexpected error in get_pdf_from_s3: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=ErrorDetail(
                status_code="500",
                error_messages=["Failed to fetch PDF from S3"]
            ).dict()
        )
    except Exception as e:
        logger.error(f"Error fetching PDF from S3: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=ErrorDetail(
                status_code="500",
                error_messages=["Failed to fetch PDF from S3"]
            ).dict()
        )

def extract_text_from_pdf(pdf_file: bytes) -> Tuple[List[str], int]:
    """Extract text from PDF file and return chunks of text along with total pages"""
    try:
        pdf_reader = PyPDF2.PdfReader(io.BytesIO(pdf_file))
        total_pages = len(pdf_reader.pages)
        pages_to_process = min(total_pages, MAX_PAGES)
        
        chunks = []
        for page_num in range(pages_to_process):
            page = pdf_reader.pages[page_num]
            text = page.extract_text()
            if text.strip():
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
async def summarize_pdf(input_data: S3URLInput) -> SummaryResponse:
    """Endpoint to process and summarize PDF content from S3"""
    start_time = time.time()
    
    try:
        # Fetch PDF from S3
        pdf_content = await get_pdf_from_s3(input_data.url)
        chunks, total_pages = extract_text_from_pdf(pdf_content)
        
        if not chunks:
            raise HTTPException(
                status_code=400,
                detail=ErrorDetail(
                    status_code="400",
                    error_messages=["No readable text found in PDF"]
                ).dict()
            )
        
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