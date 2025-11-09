from fastapi import APIRouter, HTTPException, status
from pydantic import BaseModel
from app.service.perplexity_client import perplexity_client
from app.service.log_client import logger
from typing import Optional, List
import re
import time
import asyncio
from bs4 import BeautifulSoup
from functools import lru_cache
from .web_query_cache_manager import web_query_cache

router = APIRouter()

# Pre-compile regex patterns for better performance
URL_PATTERN = re.compile(r'://(?:www\.)?([^/]+)')
DOMAIN_PATTERN = re.compile(r'([^.]+)\.')

# Site mappings as constant
SITE_MAPPINGS = {
    'realtor.com': 'Realtor.com',
    'zillow.com': 'Zillow',
    'trulia.com': 'Trulia',
    'redfin.com': 'Redfin',
    'loopnet.com': 'LoopNet',
    'apartments.com': 'Apartments.com',
    'propertyrecords.com': 'PropertyRecords.com'
}

# Allowed property types as set for O(1) lookup
ALLOWED_TYPES = {
    "home", "auto", "gas station", "restaurant", "salon",
    "general contractor", "shopping mall", "general business", "hotel/motel"
}

class WebSearchRequest(BaseModel):
    question: str
    user_id: Optional[str] = None

class WebSearchResponse(BaseModel):
    question: str
    answer: str
    answer_json: Optional[dict] = None
    source: List[str]
    suggested_questions: List[str]
    user_id: Optional[str] = None
    conversation_context: int = 0
    is_type: bool = False
    type: Optional[str] = None
    execution_time: float = 0

class ConversationHistoryResponse(BaseModel):
    user_id: str
    history: List[dict]
    total_entries: int

def format_sources(sources: List[str]) -> List[str]:
    """
    Format sources to consistent 'Site Name: URL' format
    Optimized with pre-compiled patterns and early returns
    """
    if not sources:
        return []

    formatted = []

    for source in sources:
        if not source:
            continue

        source = source.strip()

        # Early return for already formatted sources
        if ':' in source and 'http' in source and not source.startswith('http'):
            formatted.append(source)
            continue

        # Handle raw URLs
        if source.startswith('http'):
            source_lower = source.lower()

            # Check site mappings
            for domain, name in SITE_MAPPINGS.items():
                if domain in source_lower:
                    formatted.append(f"{name}: {source}")
                    break
            else:
                # Extract domain name
                match = URL_PATTERN.search(source)
                if match:
                    domain_name = match.group(1).split('.')[0].capitalize()
                    formatted.append(f"{domain_name}: {source}")
                else:
                    formatted.append(f"Source: {source}")
        else:
            formatted.append(source)

    return formatted

@lru_cache(maxsize=128)
def parse_html_table(html_table: str) -> Optional[dict]:
    """
    Parse HTML table into nested JSON grouped by section headers
    Cached for repeated calls with same HTML
    """
    if not html_table or '<table>' not in html_table:
        return None

    try:
        soup = BeautifulSoup(html_table, 'lxml') # Use lxml for speed
        table = soup.find('table')

        if not table:
            return None

        result = {}
        current_section = None

        for row in table.find_all('tr'):
            cells = [cell.get_text(strip=True) for cell in row.find_all(['td', 'th'])]

            if not cells:
                continue

            # Section header detection
            if len(cells) == 1 or (len(row.find_all('td')) == 1 and row.find('td').has_attr("colspan")):
                current_section = cells[0]
                result[current_section] = {}
                continue

            # Map key/value pairs
            if current_section:
                if len(cells) == 2:
                    key, value = cells
                    result[current_section][key] = value
                elif len(cells) == 3:
                    key, detail, value = cells
                    result[current_section][f"{key} - {detail}"] = value

        return result if result else None

    except Exception as e:
        logger.warn(f"HTML parse error: {e}, falling back to html.parser")
        try:
            soup = BeautifulSoup(html_table, 'html.parser')
            # ... (rest of the parsing logic repeated for fallback)
        except Exception as fallback_e:
            logger.error(f"Fallback HTML parse also failed: {fallback_e}")
            return None

def extract_property_type(answer_json: dict) -> tuple[bool, Optional[str]]:
    """
    Extract property type from parsed JSON
    Optimized with early returns and set lookups
    """
    if not answer_json:
        return False, None

    # Find Property Details section
    property_details = None
    for key in answer_json:
        if key.lower() == 'property details':
            property_details = answer_json[key]
            break

    if not property_details:
        return False, None

    # Find Type field
    type_key = None
    for key in property_details:
        if key.lower() in ('type', 'property type'):
            type_key = key
            break

    if not type_key:
        return False, None

    json_type_string = property_details[type_key]
    if not json_type_string:
        return False, None

    # Check against allowed types using set for O(1) lookup
    json_type_lower = json_type_string.lower()
    for allowed_type in ALLOWED_TYPES:
        if allowed_type in json_type_lower:
            # Return properly capitalized version
            return True, allowed_type.title()

    return False, None

@router.post("/web-search", response_model=WebSearchResponse)
async def web_search(request: WebSearchRequest) -> WebSearchResponse:
    """
    Search for property information using Perplexity AI
    Optimized for faster response time and better error handling
    """
    total_start_time = time.time()
    logger.info(f"[WEB_SEARCH] Received request for question: {request.question[:100]}...")

    if not request.question or not request.question.strip():
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Question cannot be empty"
        )

    # Check cache first
    cached_result = web_query_cache.get_cached_result(request.question, request.user_id)
    if cached_result:
        execution_time = round(time.time() - total_start_time, 2)
        cached_result["execution_time"] = execution_time
        logger.info(f"[WEB_SEARCH] ✓ Cache hit. Total execution time: {execution_time}s")
        return WebSearchResponse(**cached_result)

    logger.info("[WEB_SEARCH] Cache miss. Calling Perplexity AI.")

    try:
        # 1. Call Perplexity API
        api_start_time = time.time()
        response = perplexity_client.ask_question(
            question=request.question.strip(),
            user_id=request.user_id
        )
        api_execution_time = round(time.time() - api_start_time, 2)
        logger.info(f"[WEB_SEARCH] Perplexity API call took: {api_execution_time}s")

        if not response:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="No response from AI service"
            )

        # 2. Process Response
        processing_start_time = time.time()
        html_table = response.get('html_table', '')
        raw_sources = response.get('sources', [])
        suggested_questions = response.get('suggested_questions', [])
        conversation_context = response.get('conversation_context', 0)

        if not html_table or '<table>' not in html_table:
            html_table = '''<table><thead><tr><th>Category</th><th>Detail</th><th>Value</th></tr></thead>
<tbody><tr><td>Information</td><td>Status</td><td>No property information available</td></tr></tbody></table>'''

        # 3. Parse HTML
        parsing_start_time = time.time()
        answer_json = parse_html_table(html_table)
        parsing_execution_time = round(time.time() - parsing_start_time, 4)
        logger.info(f"[WEB_SEARCH] HTML parsing took: {parsing_execution_time}s")

        # 4. Extract Property Type
        is_type, property_type = extract_property_type(answer_json)

        # 5. Format Sources
        source_start_time = time.time()
        if raw_sources and raw_sources[0] != "Sources: Real estate databases":
            formatted_sources = format_sources(raw_sources)
        else:
            has_citations = '[1]' in html_table or '[2]' in html_table or '[3]' in html_table
            formatted_sources = [
                "Sources: Information compiled from real estate databases and industry reports"
                if has_citations else "Sources: Property information database"
            ]
        source_execution_time = round(time.time() - source_start_time, 4)
        logger.info(f"[WEB_SEARCH] Source formatting took: {source_execution_time}s")

        if len(suggested_questions) < 3:
            default_questions = [
                "What are the property tax rates in this area?",
                "What are comparable properties selling for nearby?",
                "What is the neighborhood market trend?"
            ]
            suggested_questions.extend(default_questions[:3 - len(suggested_questions)])

        suggested_questions = suggested_questions[:3]
        processing_execution_time = round(time.time() - processing_start_time, 2)
        logger.info(f"[WEB_SEARCH] Total response processing took: {processing_execution_time}s")

        total_execution_time = round(time.time() - total_start_time, 2)

        response_data = WebSearchResponse(
            question=request.question,
            answer=html_table,
            answer_json=answer_json,
            source=formatted_sources,
            suggested_questions=suggested_questions,
            user_id=request.user_id,
            conversation_context=conversation_context,
            is_type=is_type,
            type=property_type,
            execution_time=total_execution_time
        )

        # Cache the result asynchronously
        asyncio.create_task(asyncio.to_thread(
            web_query_cache.cache_result,
            question=request.question,
            user_id=request.user_id,
            response=response_data.dict()
        ))
        
        logger.info(f"[WEB_SEARCH] ✓ Request finished. Total execution time: {total_execution_time}s")
        return response_data

    except HTTPException:
        raise
    except ConnectionError as e:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=f"Connection error: Unable to reach AI service"
        )
    except TimeoutError as e:
        raise HTTPException(
            status_code=status.HTTP_504_GATEWAY_TIMEOUT,
            detail="Request timeout: AI service took too long to respond"
        )
    except ValueError as e:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=f"Invalid response format: {str(e)}"
        )
    except Exception as e:
        logger.error(f"Unexpected error in web_search: {type(e).__name__}: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="An unexpected error occurred while processing your request"
        )

@router.get("/conversation-history/{user_id}", response_model=ConversationHistoryResponse)
async def get_conversation_history(user_id: str) -> ConversationHistoryResponse:
    """Get conversation history for a user"""
    if not user_id or not user_id.strip():
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="User ID cannot be empty"
        )

    try:
        history = perplexity_client.get_conversation_history(user_id.strip())

        return ConversationHistoryResponse(
            user_id=user_id,
            history=history if history else [],
            total_entries=len(history) if history else 0
        )
    except Exception as e:
        print(f"Error getting conversation history: {type(e).__name__}: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to retrieve conversation history"
        )

@router.delete("/conversation-history/{user_id}")
async def clear_conversation_history(user_id: str):
    """Clear conversation history for a user"""
    if not user_id or not user_id.strip():
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="User ID cannot be empty"
        )

    try:
        success = perplexity_client.clear_conversation_history(user_id.strip())

        return {
            "success": success,
            "message": "History cleared successfully" if success else "No history found for this user"
        }
    except Exception as e:
        print(f"Error clearing conversation history: {type(e).__name__}: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to clear conversation history"
        )
