from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from app.service.perplexity_client import perplexity_client
from typing import Optional, List
import re
from bs4 import BeautifulSoup

router = APIRouter()

class WebSearchRequest(BaseModel):
    question: str
    user_id: Optional[str] = None

class WebSearchResponse(BaseModel):
    question: str
    table: str
    answer_json: Optional[dict] = None
    source: List[str]
    suggested_questions: List[str]
    user_id: Optional[str] = None
    conversation_context: int = 0
    is_type: bool = False
    type: Optional[str] = None

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

def parse_html_table(html_table: str) -> Optional[dict]:
    """Parse HTML table into a nested JSON grouped by section headers"""
    if not html_table:
        return None

    soup = BeautifulSoup(html_table, 'html.parser')
    table = soup.find('table')
    if not table:
        return None

    result = {}
    current_section = None

    for row in table.find_all('tr'):
        cells = [cell.get_text(strip=True) for cell in row.find_all(['td', 'th'])]
        
        if not cells:
            continue

        # If row has a single cell spanning columns → section header
        if len(cells) == 1 or (len(row.find_all('td')) == 1 and row.find('td').has_attr("colspan")):
            current_section = cells[0]
            result[current_section] = {}
            continue

        # If inside a section, map key/value pairs
        if current_section:
            if len(cells) == 2:
                key, value = cells
                result[current_section][key] = value
            elif len(cells) == 3:
                # In case of "Category | Detail | Value" style
                key, detail, value = cells
                result[current_section][f"{key} - {detail}"] = value

    return result if result else None

@router.post("/web-search")
async def web_search(request: WebSearchRequest) -> WebSearchResponse:
    """
    Search for property information using Perplexity AI
    
    Args:
        request: WebSearchRequest with question and optional user_id
        
    Returns:
        WebSearchResponse with HTML table, sources, suggested questions, and AI-determined property type
    """
    try:
        # Validate input
        if not request.question.strip():
            raise HTTPException(status_code=400, detail="Question cannot be empty")
        
        # Get AI response (AI will determine is_type and type directly)
        response = perplexity_client.ask_question(
            question=request.question,
            user_id=request.user_id
        )
        
        # Extract and validate response data
        html_table = response.get('html_table', '')
        raw_sources = response.get('sources', [])
        suggested_questions = response.get('suggested_questions', [])
        conversation_context = response.get('conversation_context', 0)
        
        # Initialize type variables
        is_type = False
        property_type = None
        
        # Ensure valid HTML table
        if not html_table or '<table>' not in html_table:
            html_table = '<table><thead><tr><th>Category</th><th>Detail</th><th>Value</th></tr></thead><tbody><tr><td>Information</td><td>Status</td><td>No property information available</td></tr></tbody></table>'
        
        # Parse HTML table to JSON
        answer_json = parse_html_table(html_table)
        
        # Check for type in parsed JSON and override if present
        if answer_json:
            property_details_section = None
            for key in answer_json:
                if key.lower() == 'property details':
                    property_details_section = answer_json[key]
                    break
            
            if property_details_section:
                type_key = None
                for key in property_details_section:
                    if key.lower() in ['type', 'property type']:
                        type_key = key
                        break
                
                if type_key:
                    json_type_string = property_details_section[type_key]
                    if json_type_string:
                        allowed_types = ["Home", "Auto", "Gas Station", "Restaurant", "Salon", "General Contractor", "Shopping Mall", "General Business", "Hotel/Motel"]
                        for t in allowed_types:
                            if t.lower() in json_type_string.lower():
                                property_type = t
                                is_type = True
                                break
        
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
            table=html_table,
            answer_json=answer_json,
            source=formatted_sources,
            suggested_questions=suggested_questions,
            user_id=request.user_id,
            conversation_context=conversation_context,
            is_type=is_type,
            type=property_type
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