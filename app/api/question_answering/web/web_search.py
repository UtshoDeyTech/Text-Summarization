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
        
        # Get answer from Perplexity AI (using fixed model)
        ai_response = await perplexity_client.ask_question(
            question=request.question
        )
        
        # Parse the JSON response from AI
        try:
            # Clean up any potential formatting issues
            ai_response = ai_response.strip()
            
            # Remove markdown JSON formatting if present
            if ai_response.startswith('```json'):
                ai_response = ai_response.replace('```json', '').replace('```', '').strip()
            elif ai_response.startswith('```'):
                ai_response = ai_response.replace('```', '').strip()
            
            # Find JSON object boundaries
            start_idx = ai_response.find('{')
            end_idx = ai_response.rfind('}')
            
            if start_idx == -1 or end_idx == -1:
                raise ValueError("No valid JSON object found in AI response")
            
            # Extract just the JSON part
            json_str = ai_response[start_idx:end_idx + 1]
            
            parsed_response = json.loads(json_str)
            
            # Validate response structure
            if 'answer' not in parsed_response or 'source' not in parsed_response:
                raise ValueError("AI response missing required fields: answer or source")
            
            # Validate field types
            if not isinstance(parsed_response['answer'], str):
                raise ValueError("Answer field must be a string")
            
            if not isinstance(parsed_response['source'], list):
                raise ValueError("Source field must be a list")
            
            # Ensure at least 2 sources (but don't fail if less, just warn)
            if len(parsed_response['source']) < 2:
                # Add a fallback source if needed
                while len(parsed_response['source']) < 2:
                    parsed_response['source'].append("Additional research recommended for complete verification")
            
            return WebSearchResponse(
                question=request.question,
                answer=parsed_response['answer'],
                source=parsed_response['source']
            )
            
        except json.JSONDecodeError as e:
            raise HTTPException(
                status_code=500,
                detail=f"Invalid JSON response from AI: {str(e)}"
            )
        except ValueError as e:
            raise HTTPException(
                status_code=500,
                detail=f"AI response validation error: {str(e)}"
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