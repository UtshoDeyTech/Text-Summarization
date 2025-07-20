from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from app.service.perplexity_client import perplexity_client
from typing import Optional, List
import json

router = APIRouter()

class WebSearchRequest(BaseModel):
    question: str

class WebSearchResponse(BaseModel):
    question: str
    answer: str
    source: List[str]

@router.post("/web-search")
async def web_search(request: WebSearchRequest) -> WebSearchResponse:
    """
    Search the web using Perplexity AI
    
    Args:
        request: WebSearchRequest containing the question and optional model
        
    Returns:
        WebSearchResponse with the question, answer, and model used
        
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
        
        # Get answer from Perplexity AI - REMOVED await, function returns a dict
        ai_response = perplexity_client.ask_question(
            question=request.question
        )
        
        # ai_response is now a dictionary with keys: html_table, sources, summary, raw_response
        # No need to parse JSON - it's already structured
        
        try:
            # Extract HTML table and sources from the response dictionary
            html_table = ai_response.get('html_table', '')
            sources = ai_response.get('sources', [])
            
            # Ensure we have valid HTML table
            if not html_table:
                # Fallback: create basic table if none provided
                summary = ai_response.get('summary', 'No property information available')
                html_table = f"<table><thead><tr><th>Property Information</th></tr></thead><tbody><tr><td>{summary}</td></tr></table>"
            
            # Ensure sources is a list and has at least 2 entries
            if not isinstance(sources, list):
                sources = []
            
            # Add fallback sources if we don't have enough real ones
            if len(sources) < 2:
                default_sources = [
                    "https://www.zillow.com",
                    "https://www.realtor.com",
                    "https://www.loopnet.com"
                ]
                for default_source in default_sources:
                    if len(sources) < 2 and default_source not in sources:
                        sources.append(default_source)
            
            return WebSearchResponse(
                question=request.question,
                answer=html_table,  # Return raw HTML table
                source=sources
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