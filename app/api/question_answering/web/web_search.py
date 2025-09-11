from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from app.service.perplexity_client import perplexity_client
from typing import Optional, List
import re

router = APIRouter()

class WebSearchRequest(BaseModel):
    question: str
    user_id: Optional[str] = None

class WebSearchResponse(BaseModel):
    question: str
    answer: str
    source: List[str]
    suggested_questions: List[str]
    user_id: Optional[str] = None
    conversation_context: int = 0

class ConversationHistoryResponse(BaseModel):
    user_id: str
    history: List[dict]
    total_entries: int

def format_sources(sources: List[str]) -> List[str]:
    """Format sources to consistent 'Site Name: URL' format"""
    formatted = []
    
    site_mappings = {
        'realtor.com': 'Realtor.com',
        'zillow.com': 'Zillow',
        'trulia.com': 'Trulia',
        'redfin.com': 'Redfin',
        'loopnet.com': 'LoopNet',
        'apartments.com': 'Apartments.com',
        'propertyrecords.com': 'PropertyRecords.com'
    }
    
    for source in sources:
        source = source.strip()
        
        # Already formatted
        if ':' in source and 'http' in source and not source.startswith('http'):
            formatted.append(source)
            continue
        
        # Raw URL
        if source.startswith('http'):
            for domain, name in site_mappings.items():
                if domain in source.lower():
                    formatted.append(f"{name}: {source}")
                    break
            else:
                # Unknown domain
                domain_match = re.search(r'://(?:www\.)?([^/]+)', source)
                if domain_match:
                    domain_name = domain_match.group(1).split('.')[0].capitalize()
                    formatted.append(f"{domain_name}: {source}")
                else:
                    formatted.append(f"Source: {source}")
        else:
            formatted.append(source)
    
    return formatted

@router.post("/web-search")
async def web_search(request: WebSearchRequest) -> WebSearchResponse:
    """
    Search for property information using Perplexity AI
    
    Args:
        request: WebSearchRequest with question and optional user_id
        
    Returns:
        WebSearchResponse with HTML table, sources, and suggested questions
    """
    try:
        # Validate input
        if not request.question.strip():
            raise HTTPException(status_code=400, detail="Question cannot be empty")
        
        # Get AI response
        response = perplexity_client.ask_question(
            question=request.question,
            user_id=request.user_id
        )
        
        # Extract and validate response data
        html_table = response.get('html_table', '')
        raw_sources = response.get('sources', [])
        suggested_questions = response.get('suggested_questions', [])
        conversation_context = response.get('conversation_context', 0)
        
        # Ensure valid HTML table
        if not html_table or '<table>' not in html_table:
            html_table = '<table><thead><tr><th>Category</th><th>Detail</th><th>Value</th></tr></thead><tbody><tr><td>Information</td><td>Status</td><td>No property information available</td></tr></tbody></table>'
        
        # Format sources - ensure we always have sources
        if raw_sources and len(raw_sources) > 0 and raw_sources[0] != "Sources: Real estate databases":
            formatted_sources = format_sources(raw_sources)
        else:
            # If no real sources found, check if content has citations
            if '[1]' in html_table or '[2]' in html_table or '[3]' in html_table:
                formatted_sources = ["Sources: Information compiled from real estate databases and industry reports"]
            else:
                formatted_sources = ["Sources: Property information database"]
        
        # Ensure exactly 3 suggested questions
        if len(suggested_questions) < 3:
            default_questions = [
                "What are the property tax rates in this area?",
                "What are comparable properties selling for nearby?",
                "What is the neighborhood market trend?"
            ]
            suggested_questions.extend(default_questions[:3-len(suggested_questions)])
        
        suggested_questions = suggested_questions[:3]
        
        return WebSearchResponse(
            question=request.question,
            answer=html_table,
            source=formatted_sources,
            suggested_questions=suggested_questions,
            user_id=request.user_id,
            conversation_context=conversation_context
        )
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Search failed: {str(e)}")

@router.get("/conversation-history/{user_id}")
async def get_conversation_history(user_id: str) -> ConversationHistoryResponse:
    """Get conversation history for a user"""
    try:
        history = perplexity_client.get_conversation_history(user_id)
        return ConversationHistoryResponse(
            user_id=user_id,
            history=history,
            total_entries=len(history)
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get history: {str(e)}")

@router.delete("/conversation-history/{user_id}")
async def clear_conversation_history(user_id: str):
    """Clear conversation history for a user"""
    try:
        success = perplexity_client.clear_conversation_history(user_id)
        return {"success": success, "message": "History cleared" if success else "No history found"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to clear history: {str(e)}")