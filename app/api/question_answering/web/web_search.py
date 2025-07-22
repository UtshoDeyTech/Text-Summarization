from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from app.service.perplexity_client import perplexity_client
from typing import Optional, List
import json
import re

router = APIRouter()

class WebSearchRequest(BaseModel):
    question: str

class WebSearchResponse(BaseModel):
    question: str
    answer: str
    source: List[str]
    suggested_questions: List[str]

def format_sources_properly(sources: List[str]) -> List[str]:
    """
    Format sources from raw URLs to 'Site Name: URL' format
    """
    formatted_sources = []
    
    for source in sources:
        source = source.strip()
        
        # If already in correct format "Site: URL", keep it
        if ':' in source and 'http' in source and not source.startswith('http'):
            formatted_sources.append(source)
            continue
        
        # If it's a raw URL, format it
        if source.startswith('http'):
            formatted_source = format_url_to_source(source)
            formatted_sources.append(formatted_source)
        else:
            # Keep other formats as-is
            formatted_sources.append(source)
    
    return formatted_sources

def format_url_to_source(url: str) -> str:
    """
    Convert a raw URL to 'Site Name: URL' format
    """
    url = url.strip()
    
    # Define site mappings
    site_mappings = {
        'realtor.com': 'Realtor.com',
        'zillow.com': 'Zillow',
        'trulia.com': 'Trulia',
        'redfin.com': 'Redfin',
        'loopnet.com': 'LoopNet',
        'apartments.com': 'Apartments.com',
        'rentals.com': 'Rentals.com',
        'propertyrecords.com': 'PropertyRecords.com',
        'cityfeet.com': 'CityFeet',
        'myelisting.com': 'MyeListing',
        '7-eleven.com': '7-Eleven',
        'homes.com': 'Homes.com',
        'propertyshark.com': 'PropertyShark',
        'crexi.com': 'Crexi',
        'showcase.com': 'Showcase',
        'rocketmls.com': 'RocketMLS',
        'homesnap.com': 'Homesnap',
        'landwatch.com': 'LandWatch',
        'landquest.com': 'Land Quest',
        'commercialcafe.com': 'CommercialCafe'
    }
    
    # Check for known sites
    for domain, site_name in site_mappings.items():
        if domain in url.lower():
            return f"{site_name}: {url}"
    
    # For unknown sites, extract domain name
    domain_match = re.search(r'://(?:www\.)?([^/]+)', url)
    if domain_match:
        domain = domain_match.group(1)
        # Remove common TLD and get the main part
        main_domain = domain.split('.')[0]
        # Capitalize first letter
        site_name = main_domain.capitalize()
        return f"{site_name}: {url}"
    else:
        return f"Source: {url}"

@router.post("/web-search")
async def web_search(request: WebSearchRequest) -> WebSearchResponse:
    """
    Search the web using Perplexity AI
    
    Args:
        request: WebSearchRequest containing the question and optional model
        
    Returns:
        WebSearchResponse with the question, answer, formatted sources, and suggested questions
        
    Raises:
        HTTPException: If the search request fails
    """
    try:
        # Validate input
        if not request.question.strip():
            raise HTTPException(
                status_code=400,
                detail="Question cannot be empty"
            )
        
        # Get answer from Perplexity AI
        ai_response = perplexity_client.ask_question(
            question=request.question
        )
        
        try:
            # Extract HTML table, sources, and suggested questions from the response dictionary
            html_table = ai_response.get('html_table', '')
            raw_sources = ai_response.get('sources', [])
            suggested_questions = ai_response.get('suggested_questions', [])
            
            # Ensure we have valid HTML table
            if not html_table:
                # Fallback: create basic table if none provided
                summary = ai_response.get('summary', 'No property information available')
                html_table = f"<table><thead><tr><th>Property Information</th></tr></thead><tbody><tr><td>{summary}</td></tr></table>"
            
            # Ensure sources is a list
            if not isinstance(raw_sources, list):
                raw_sources = []
            
            # Ensure suggested_questions is a list and has exactly 3 questions
            if not isinstance(suggested_questions, list):
                suggested_questions = []
            
            # Ensure we have exactly 3 suggested questions
            while len(suggested_questions) < 3:
                suggested_questions.append("What are additional details about this property?")
            
            # Trim to exactly 3 questions if we have more
            suggested_questions = suggested_questions[:3]
            
            # Format sources to the desired "Site Name: URL" format
            formatted_sources = format_sources_properly(raw_sources)
            
            # If no sources found, try to extract from the HTML table content
            if not formatted_sources:
                # Look for citation numbers [1], [2], etc. in the HTML table
                citation_pattern = r'\[(\d+)\]'
                citations_found = re.findall(citation_pattern, html_table)
                
                if citations_found:
                    # Add a generic source note if citations exist but no URLs found
                    formatted_sources = ["Sources: Information compiled from real estate databases and listings"]
                else:
                    # No citations or sources found
                    formatted_sources = ["Sources: Property information database"]
            
            return WebSearchResponse(
                question=request.question,
                answer=html_table,
                source=formatted_sources,
                suggested_questions=suggested_questions
            )
            
        except Exception as e:
            raise HTTPException(
                status_code=500,
                detail=f"Response processing error: {str(e)}"
            )
        
    except Exception as e:
        # Check if it's already an HTTPException
        if isinstance(e, HTTPException):
            raise
        
        # Handle other exceptions
        raise HTTPException(
            status_code=500,
            detail=f"Web search failed: {str(e)}"
        )

# Alternative endpoint for testing source formatting and suggested questions
@router.post("/web-search-debug")
async def web_search_debug(request: WebSearchRequest) -> dict:
    """
    Debug endpoint to see raw and formatted sources plus suggested questions
    """
    try:
        ai_response = perplexity_client.ask_question(request.question)
        
        raw_sources = ai_response.get('sources', [])
        formatted_sources = format_sources_properly(raw_sources)
        suggested_questions = ai_response.get('suggested_questions', [])
        
        return {
            "question": request.question,
            "raw_sources": raw_sources,
            "formatted_sources": formatted_sources,
            "suggested_questions": suggested_questions,
            "html_table": ai_response.get('html_table', ''),
            "full_response": ai_response
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))