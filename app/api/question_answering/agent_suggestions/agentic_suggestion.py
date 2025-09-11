from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field
from openai import AsyncOpenAI
import json
from typing import Dict, Optional
from datetime import datetime
from config import OPENAI_API_KEY

router = APIRouter()
client = AsyncOpenAI(api_key=OPENAI_API_KEY)

class PreviousStages(BaseModel):
    stage_1_date: Optional[str] = ""
    stage_2_date: Optional[str] = ""
    stage_3_date: Optional[str] = ""

class LeadRequest(BaseModel):
    lead_name: str = Field(..., description="Name of the lead/client")
    lead_type: str = Field(..., description="Type of insurance (Auto, Home, Life, etc.)")
    progress: int = Field(..., ge=0, le=100, description="Profile completion percentage")
    latest_comment: str = Field(..., description="Latest system comment about the lead")
    current_stage: int = Field(..., ge=1, le=4, description="Current stage (1-4)")
    current_stage_date: str = Field(..., description="Date of current stage")
    previous_stages: Optional[Dict] = Field(default={}, description="Previous stage dates")

class AgenticResponse(BaseModel):
    immediate_plan: str
    secondary_plan: str
    lead_info: Dict
    timestamp: str

class AgenticSuggestionGenerator:
    def __init__(self):
        """Initialize the Agentic Suggestion Generator."""
        self.stages = {
            1: "Lead Processing",
            2: "Quote Processing", 
            3: "Client Approval",
            4: "Policy Bind"
        }
    
    def format_lead_data(self, lead_data: Dict) -> Dict:
        """Format and validate lead data structure."""
        formatted_data = {
            "lead_name": lead_data.get("lead_name", ""),
            "lead_type": lead_data.get("lead_type", ""),
            "progress_percentage": lead_data.get("progress", 0),
            "latest_comment": lead_data.get("latest_comment", ""),
            "current_stage": lead_data.get("current_stage", 1),
            "current_stage_date": lead_data.get("current_stage_date", ""),
            "previous_stages": self._format_previous_stages(
                lead_data.get("current_stage", 1),
                lead_data.get("previous_stages", {})
            )
        }
        return formatted_data
    
    def _format_previous_stages(self, current_stage: int, previous_stages: Dict) -> Dict:
        """Format previous stages based on current stage logic."""
        if current_stage == 1:
            return {}
        elif current_stage == 2:
            return {"stage_1_date": previous_stages.get("stage_1_date", "")}
        elif current_stage == 3:
            return {
                "stage_1_date": previous_stages.get("stage_1_date", ""),
                "stage_2_date": previous_stages.get("stage_2_date", "")
            }
        else:  # stage 4 or higher
            return {
                "stage_1_date": previous_stages.get("stage_1_date", ""),
                "stage_2_date": previous_stages.get("stage_2_date", ""),
                "stage_3_date": previous_stages.get("stage_3_date", "")
            }
    
    def create_prompt(self, lead_data: Dict) -> str:
        """Create a focused prompt for OpenAI based on lead data."""
        current_stage_name = self.stages.get(lead_data["current_stage"], "Unknown Stage")
        
        prompt = f"""You are an expert sales consultant specializing in insurance policy sales. Analyze the following lead data and provide the best approach for an agent to move this client to the final "Policy Bind" stage.

LEAD INFORMATION:
- Lead Name: {lead_data['lead_name']}
- Lead Type: {lead_data['lead_type']}
- Profile Completion: {lead_data['progress_percentage']}%
- Latest System Comment: {lead_data['latest_comment']}
- Current Stage: {current_stage_name} (Stage {lead_data['current_stage']})
- Current Stage Date: {lead_data['current_stage_date']}
- Previous Stages: {json.dumps(lead_data['previous_stages'], indent=2)}

SALES FUNNEL STAGES:
1. Lead Processing - Form submitted, lead created
2. Quote Processing - Agent processes specific data to complete this stage
3. Client Approval - Customer needs to approve the proposal
4. Policy Bind - Final stage after payment (TARGET GOAL)

You must return a valid JSON object with exactly these two fields:
{{
    "immediate_plan": "The most important action the agent should take right now",
    "secondary_plan": "Follow-up action or backup approach if immediate plan doesn't work"
}}

Requirements:
1. Response must be a valid JSON object
2. Both fields must be present and non-empty
3. Keep each plan concise but specific and actionable
4. Tailor recommendations to the current stage and lead information
5. Focus on moving the lead to the final "Policy Bind" stage
6. No additional fields or formatting"""
        
        return prompt

async def generate_agentic_suggestion(lead_data: Dict) -> str:
    """Generate agentic suggestions using OpenAI"""
    generator = AgenticSuggestionGenerator()
    formatted_data = generator.format_lead_data(lead_data)
    prompt = generator.create_prompt(formatted_data)
    
    try:
        response = await client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {
                    "role": "system", 
                    "content": "You are an expert sales consultant with deep knowledge of insurance sales processes and client psychology. Always respond with valid JSON format."
                },
                {
                    "role": "user", 
                    "content": prompt
                }
            ],
            temperature=0.7,
            max_tokens=1000,
            response_format={"type": "json_object"}  # Ensure JSON response
        )
        return response.choices[0].message.content
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error generating agentic suggestion: {str(e)}"
        )

@router.post("/generate-suggestion")
async def generate_suggestion(request: LeadRequest) -> AgenticResponse:
    """
    Generate agentic suggestions for lead approach.
    
    This endpoint takes lead information and generates AI-powered suggestions
    for how agents should approach the client to move them through the sales funnel.
    """
    try:
        # Convert request to dictionary
        lead_data = {
            "lead_name": request.lead_name,
            "lead_type": request.lead_type,
            "progress": request.progress,
            "latest_comment": request.latest_comment,
            "current_stage": request.current_stage,
            "current_stage_date": request.current_stage_date,
            "previous_stages": request.previous_stages
        }
        
        # Generate agentic suggestions
        suggestion_json = await generate_agentic_suggestion(lead_data)
        
        # Parse the JSON string response from OpenAI
        try:
            # Clean up any potential formatting issues
            suggestion_json = suggestion_json.strip()
            if not suggestion_json.startswith('{'):
                raise ValueError("Response is not in JSON format")
                
            suggestion_content = json.loads(suggestion_json)
            
            # Validate response structure
            required_fields = {'immediate_plan', 'secondary_plan'}
            missing_fields = required_fields - set(suggestion_content.keys())
            if missing_fields:
                raise ValueError(f"Missing required fields: {', '.join(missing_fields)}")
            
            # Validate field content
            for field in required_fields:
                if not isinstance(suggestion_content[field], str) or not suggestion_content[field].strip():
                    raise ValueError(f"Field '{field}' must be a non-empty string")
            
            # Format lead info for response
            generator = AgenticSuggestionGenerator()
            formatted_lead_data = generator.format_lead_data(lead_data)
            
            return AgenticResponse(
                immediate_plan=suggestion_content['immediate_plan'],
                secondary_plan=suggestion_content['secondary_plan'],
                lead_info={
                    "lead_name": formatted_lead_data['lead_name'],
                    "lead_type": formatted_lead_data['lead_type'],
                    "current_stage": formatted_lead_data['current_stage'],
                    "current_stage_name": generator.stages.get(formatted_lead_data['current_stage'], "Unknown"),
                    "progress_percentage": formatted_lead_data['progress_percentage']
                },
                timestamp=datetime.now().isoformat()
            )
            
        except json.JSONDecodeError as e:
            raise HTTPException(
                status_code=500,
                detail=f"Invalid JSON response from AI: {str(e)}"
            )
        except ValueError as e:
            raise HTTPException(
                status_code=500,
                detail=str(e)
            )
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error processing suggestion request: {str(e)}"
        )

@router.post("/batch-generate-suggestions")
async def batch_generate_suggestions(requests: list[LeadRequest]) -> list[AgenticResponse]:
    """
    Generate agentic suggestions for multiple leads in batch.
    
    This endpoint processes multiple leads at once and returns
    suggestions for each lead.
    """
    if len(requests) > 50:  # Limit batch size
        raise HTTPException(
            status_code=400,
            detail="Batch size cannot exceed 50 leads"
        )
    
    try:
        results = []
        for request in requests:
            try:
                # Generate suggestion for each lead
                result = await generate_suggestion(request)
                results.append(result)
            except Exception as e:
                # Add error result for failed leads
                results.append(AgenticResponse(
                    immediate_plan=f"Error processing lead: {str(e)}",
                    secondary_plan="Please retry with valid lead data",
                    lead_info={
                        "lead_name": request.lead_name,
                        "lead_type": request.lead_type,
                        "current_stage": request.current_stage,
                        "current_stage_name": "Error",
                        "progress_percentage": request.progress
                    },
                    timestamp=datetime.now().isoformat()
                ))
        
        return results
        
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error processing batch suggestions: {str(e)}"
        )