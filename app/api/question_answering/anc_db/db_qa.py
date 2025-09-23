from fastapi import APIRouter, HTTPException
from typing import List, Optional
from pydantic import BaseModel, Field, validator
from config import OPENAI_API_KEY, DB_HOST, DB_USER, DB_PASSWORD, DB_NAME, DB_PORT
from app.service.log_client import logger
from langchain_community.utilities import SQLDatabase
from langchain_openai import ChatOpenAI
from langchain_community.agent_toolkits import SQLDatabaseToolkit
from langchain_community.agent_toolkits.sql.base import create_sql_agent
from langchain.schema import SystemMessage
import time
import os
import re
from urllib.parse import quote
from langchain.agents import AgentExecutor
from langchain_core.agents import AgentAction, AgentFinish

router = APIRouter()

# Security Configuration
class SecurityConfig:
    MAX_QUERY_LENGTH = 3000
    MAX_ITERATIONS = 8
    MAX_EXECUTION_TIME = 45
    RATE_LIMIT_PER_MINUTE = 15
    
    FORBIDDEN_PATTERNS = [
        r'\b(DROP|DELETE|INSERT|UPDATE|CREATE|ALTER|TRUNCATE)\b',
        r'\b(EXEC|EXECUTE|SP_|XP_)\b',
        r'\b(LOAD_FILE|INTO\s+OUTFILE|DUMPFILE)\b',
        r'(\-\-|\#|\/\*|\*\/)',
        r'(\;.*\;)',
    ]

# Allowed tables - only leads tables and customer_master
ALLOWED_TABLES = {
    'auto_leads',
    'gasstation_leads', 
    'general_business_leads',
    'general_contractor_leads',
    'home_leads',
    'hotel_leads',
    'restaurant_leads',
    'salon_leads',
    'shopping_leads',
    'tank_leads',
    'customer_master'
}

class AncDBInput(BaseModel):
    question: str = Field(description="Any question about leads or customer data", max_length=SecurityConfig.MAX_QUERY_LENGTH)
    agent_id: Optional[str] = Field(default=None, description="Agent ID for filtering", max_length=100)
    agency_id: Optional[str] = Field(default=None, description="Agency ID for filtering", max_length=100)

    @validator('question')
    def validate_question(cls, v):
        if not v or not v.strip():
            raise ValueError("Question cannot be empty")
        
        for pattern in SecurityConfig.FORBIDDEN_PATTERNS:
            if re.search(pattern, v, re.IGNORECASE):
                logger.warning(f"Suspicious pattern detected in question: {pattern}")
        
        return v.strip()

    @validator('agent_id', 'agency_id')
    def validate_ids(cls, v):
        if v:
            if not re.match(r'^[a-zA-Z0-9_-]+$', v):
                raise ValueError("ID contains invalid characters")
        return v

class AncDBResponse(BaseModel):
    question: str
    answer: str
    sources: List[str] = []
    suggested_questions: List[str] = []
    model_used: str
    status_code: str = "200"
    found: bool = True
    execution_time: float = 0
    filtered_by: str = ""

def get_restricted_system_prompt(agent_id: Optional[str], agency_id: Optional[str]) -> str:
    """Create a system prompt that restricts access to only leads tables and customer_master"""
    
    filter_info = ""
    if agent_id and agency_id:
        filter_info = f"FILTERING: When querying tables with agent_id and agency_id columns, ALWAYS add: WHERE agent_id = '{agent_id}' AND agency_id = '{agency_id}'"
    elif agent_id:
        filter_info = f"FILTERING: When querying tables with agent_id column, ALWAYS add: WHERE agent_id = '{agent_id}'"
    elif agency_id:
        filter_info = f"FILTERING: When querying tables with agency_id column, ALWAYS add: WHERE agency_id = '{agency_id}'"
    
    allowed_tables_list = ', '.join(sorted(ALLOWED_TABLES))
    
    return f"""You are a specialized database assistant for lead and customer data ONLY.

{filter_info}

STRICT TABLE RESTRICTIONS:
You can ONLY query these tables:
- auto_leads
- gasstation_leads  
- general_business_leads
- general_contractor_leads
- home_leads
- hotel_leads
- restaurant_leads
- salon_leads
- shopping_leads
- tank_leads
- customer_master

FORBIDDEN ACTIONS:
- Do NOT query any other tables (AgencyProfile, agent_master, billing, etc.)
- Do NOT use SHOW TABLES or explore database structure
- If asked about other tables, politely decline and redirect to leads data

INSTRUCTIONS:
- Only use SELECT queries on the allowed tables above
- Apply agent_id/agency_id filtering when those columns exist
- Focus on customer information: names, emails, addresses, phone numbers
- Use LIKE '%term%' for flexible searching
- Be efficient - try the most relevant table first

EXAMPLE QUERIES:
- Emails in home leads: "SELECT email, contact_email FROM home_leads WHERE agent_id = 'value' AND email IS NOT NULL"
- Customer names: "SELECT name, first_name, last_name FROM auto_leads WHERE agency_id = 'value'"
- Addresses: "SELECT location_address, mailing_address FROM home_leads WHERE agent_id = 'value'"

RESPONSE GUIDELINES:
- If asked about non-leads tables: "I can only help with leads and customer data. Please ask about specific lead types (home, auto, restaurant, etc.) or customer information."
- Always explain which table(s) you searched
- Provide specific, useful information when found
- If no results: "No matching records found in the [table_name] table with your current filters."

Remember: You are restricted to leads tables and customer_master only!"""

class RestrictedAncDBAgent:
    def __init__(self, agent_id: Optional[str] = None, agency_id: Optional[str] = None):
        self.agent_id = agent_id
        self.agency_id = agency_id
        self.setup_environment()
        self.connection_url = self.build_connection_url()
        self.db = None
        self.agent = None
        
    def setup_environment(self):
        if not OPENAI_API_KEY:
            raise ValueError("OPENAI_API_KEY not found in environment variables")
        os.environ['OPENAI_API_KEY'] = OPENAI_API_KEY
        
    def build_connection_url(self) -> str:
        if all([DB_HOST, DB_USER, DB_PASSWORD, DB_NAME, DB_PORT]):
            logger.info(f"Using configured database: {DB_HOST}")
            encoded_password = quote(DB_PASSWORD)
            return f"mysql+pymysql://{DB_USER}:{encoded_password}@{DB_HOST}:{DB_PORT}/{DB_NAME}?charset=utf8mb4"
        else:
            logger.warning("Database config incomplete, using fallback connection")
            return "mysql+pymysql://mypolicylistdb:Anc!2024@174.138.44.203:3306/mypolicylistdb?charset=utf8mb4"
    
    def test_connection(self) -> bool:
        try:
            self.db = SQLDatabase.from_uri(self.connection_url)
            # Verify only allowed tables are accessible
            all_tables = self.db.get_usable_table_names()
            available_allowed = [t for t in all_tables if t in ALLOWED_TABLES]
            logger.info(f"Connected to database. Available allowed tables: {available_allowed}")
            return True
        except Exception as e:
            logger.error(f"Database connection failed: {e}")
            return False
    
    def initialize_agent(self) -> bool:
        try:
            self.llm = ChatOpenAI(
                model_name="gpt-4o-mini",
                temperature=0,
                max_tokens=2000,
                request_timeout=SecurityConfig.MAX_EXECUTION_TIME
            )
            
            # Create a restricted toolkit that only includes allowed tables
            toolkit = SQLDatabaseToolkit(db=self.db, llm=self.llm)
            
            system_message = SystemMessage(content=get_restricted_system_prompt(self.agent_id, self.agency_id))
            
            self.agent = create_sql_agent(
                llm=self.llm,
                toolkit=toolkit,
                verbose=True,
                agent_type="openai-tools",
                system_message=system_message,
                max_iterations=SecurityConfig.MAX_ITERATIONS,
                handle_parsing_errors=True,
                early_stopping_method="generate"
            )
            
            logger.info(f"Restricted agent initialized for agent_id: {self.agent_id}, agency_id: {self.agency_id}")
            return True
            
        except Exception as e:
            logger.error(f"Agent initialization failed: {e}")
            return False
    
    def validate_query_safety(self, query: str) -> bool:
        """Validate that query only accesses allowed tables"""
        query_upper = query.upper()
        
        # Check for forbidden table access patterns
        forbidden_tables = [
            'AGENCYPROFILE', 'BILLINGADDON', 'CHATMESSAGES', 'CHATSESSION',
            'COMPANYINFO', 'PDFDATA', 'ROLE', 'USERCOMPANYACCESS',
            'AGENT_LOGIN_ACTIVITY', 'AGENT_MASTER', 'API_REQUEST_LOG',
            'BILLING', 'CHATBOT_QUESTION', 'CHATHISTORYLOG', 'COMMENT_MASTER',
            'COUPON', 'DRIVER_DETAILS', 'EPAY_TRANSACTIONS', 'LEAD_SOURCE',
            'LEAD_WORKFLOW', 'LEAD_WORKFLOW_PAYMENT', 'LEAD_WORKFLOW_REASON',
            'MANAGE_TEAM', 'NOTIFICATION', 'PACKAGE_DESCRIPTION', 'PACKAGE_INFO',
            'PAYMENTDETAIL', 'PDF_TEMPLATES', 'RESPONSE_MESSAGE', 'SECONDARY_CONTACT',
            'TEAMMATES', 'USER_PACKAGE', 'USERS', 'VEHICLE_DETAILS', 'WORKFLOW_MASTER'
        ]
        
        for forbidden_table in forbidden_tables:
            if f' {forbidden_table}' in query_upper or f'`{forbidden_table}`' in query_upper:
                logger.warning(f"Attempted to access forbidden table: {forbidden_table}")
                return False
        
        return True

    def generate_friendly_response(self, question: str) -> tuple:
        """Generate a friendly AI response when unable to find the answer"""
        try:
            friendly_llm = ChatOpenAI(
                model_name="gpt-4o-mini",
                temperature=0.7,
                max_tokens=200
            )

            prompt = f"""The user asked: "{question}"

I was unable to find the specific information in the database after searching thoroughly.
Generate a friendly, helpful response that:
1. Acknowledges the question
2. Explains that I need more specific information to help better
3. Suggests how they can rephrase or be more specific
4. Remains professional and encouraging

Keep the response concise and under 100 words."""

            response = friendly_llm.invoke(prompt)
            friendly_answer = response.content if hasattr(response, 'content') else str(response)

            suggestions = [
                "Can you provide more specific details about what you're looking for?",
                "Try asking about a specific lead type (home, auto, restaurant, etc.)",
                "Include specific criteria like date ranges or customer details"
            ]

            return friendly_answer, suggestions

        except Exception as e:
            logger.error(f"Error generating friendly response: {e}")
            return (
                "I'm having trouble finding the specific information you're looking for. Could you please be more specific about what you'd like to know? This will help me provide you with a better response.",
                [
                    "Can you provide more specific details about what you're looking for?",
                    "Try asking about a specific lead type (home, auto, restaurant, etc.)",
                    "Include specific criteria like date ranges or customer details"
                ]
            )

    def process_question(self, question: str) -> tuple:
        try:
            logger.info(f"Processing restricted question: {question}")
            
            # Check if question is asking about forbidden tables
            question_lower = question.lower()
            forbidden_keywords = ['agency', 'billing', 'chat', 'user', 'agent_master', 'workflow']
            
            if any(keyword in question_lower for keyword in forbidden_keywords):
                if not any(allowed in question_lower for allowed in ['leads', 'customer']):
                    return (
                        "I can only help with leads and customer data. Please ask about specific lead types (home leads, auto leads, restaurant leads, etc.) or customer information.",
                        [
                            "What customer emails are in home leads?",
                            "How many auto leads do I have?", 
                            "What are the customer names in restaurant leads?"
                        ]
                    )
            
            # Create focused prompt for leads data only
            prompt = f"""
Question: {question}

Search ONLY in these allowed tables for the answer:
- auto_leads, gasstation_leads, general_business_leads, general_contractor_leads
- home_leads, hotel_leads, restaurant_leads, salon_leads, shopping_leads, tank_leads  
- customer_master

Apply the filtering rules if tables have agent_id/agency_id columns.
Focus on customer information: names, emails, addresses, phone numbers.

Format your response as:
ANSWER: [Your specific answer with data found]
SUGGESTED_QUESTIONS:
1. [Related question about leads data]
2. [Related question about customer information]
3. [Related question about other lead types]
"""
            
            try:
                response = self.agent.invoke({"input": prompt})
                result = response["output"] if isinstance(response, dict) else str(response)
            except Exception as agent_error:
                # Check if the error is related to MAX_ITERATIONS being reached or timeout
                error_str = str(agent_error).lower()
                if (
                    "maximum iterations" in error_str or
                    "max iterations" in error_str or
                    "iteration limit" in error_str or
                    "agent stopped due to iteration limit" in error_str or
                    "too many iterations" in error_str or
                    "timeout" in error_str or
                    "execution time" in error_str
                ):
                    logger.info(f"MAX_ITERATIONS or timeout reached for question: {question}")
                    return self.generate_friendly_response(question)
                else:
                    # Re-raise other types of errors
                    raise agent_error
            
            # Parse the response
            if "SUGGESTED_QUESTIONS:" in result:
                parts = result.split("SUGGESTED_QUESTIONS:")
                answer = parts[0].replace("ANSWER:", "").strip()
                suggestions = []
                if len(parts) > 1:
                    lines = parts[1].strip().split("\n")
                    for line in lines:
                        clean = re.sub(r'^\d+\.\s*', '', line.strip())
                        if clean and len(suggestions) < 3:
                            suggestions.append(clean)
            else:
                answer = result.strip()
                suggestions = []
            
            # Ensure we have 3 suggestions related to leads data
            default_suggestions = [
                "What other customer information is available in this lead type?",
                "How many total leads are in this category?",
                "What are the contact details for these customers?"
            ]
            while len(suggestions) < 3:
                suggestions.append(default_suggestions[len(suggestions)])
            
            return answer, suggestions[:3]
            
        except Exception as e:
            logger.error(f"Error processing restricted question: {e}")
            # For any other unexpected errors, use the friendly response
            return self.generate_friendly_response(question)

# Global agent management
_restricted_agents = {}

def get_restricted_agent(agent_id: Optional[str], agency_id: Optional[str]):
    """Get or create a restricted agent instance"""
    cache_key = f"restricted_{agent_id or 'none'}_{agency_id or 'none'}"
    
    if cache_key not in _restricted_agents:
        try:
            _restricted_agents[cache_key] = RestrictedAncDBAgent(agent_id=agent_id, agency_id=agency_id)
            
            if not _restricted_agents[cache_key].test_connection():
                raise HTTPException(status_code=500, detail="Database connection failed")
            
            if not _restricted_agents[cache_key].initialize_agent():
                raise HTTPException(status_code=500, detail="Agent initialization failed")
                
        except Exception as e:
            logger.error(f"Failed to create restricted agent: {str(e)}")
            raise HTTPException(status_code=500, detail=f"Agent creation failed: {str(e)}")
    
    return _restricted_agents[cache_key]

@router.post("/ask-ancdb", response_model=AncDBResponse)
async def ask_ancdb_restricted(input_data: AncDBInput):
    """Ask questions about leads and customer data only"""
    start_time = time.time()
    
    try:
        # Get the restricted agent
        agent = get_restricted_agent(input_data.agent_id, input_data.agency_id)
        
        # Process the question
        answer, suggested_questions = agent.process_question(input_data.question)
        
        # Build filter description
        filter_parts = []
        if input_data.agent_id:
            filter_parts.append(f"agent_id: {input_data.agent_id}")
        if input_data.agency_id:
            filter_parts.append(f"agency_id: {input_data.agency_id}")
        filtered_by = ", ".join(filter_parts) if filter_parts else "No filtering"
        
        execution_time = round(time.time() - start_time, 2)
        
        logger.info(f"Restricted query completed in {execution_time}s with filtering: {filtered_by}")
        
        return AncDBResponse(
            question=input_data.question,
            answer=answer,
            sources=["ANC Database (Leads & Customer Data Only)"],
            suggested_questions=suggested_questions,
            model_used="GPT-4o-mini (Restricted Mode)",
            status_code="200",
            found=True,
            execution_time=execution_time,
            filtered_by=filtered_by
        )
        
    except HTTPException:
        raise
    except Exception as e:
        execution_time = time.time() - start_time
        logger.error(f"Error in restricted endpoint: {str(e)}")
        
        return AncDBResponse(
            question=input_data.question,
            answer="I can only help with leads and customer data. Please ask about specific lead types (home, auto, restaurant, etc.) or customer information.",
            sources=[],
            suggested_questions=[
                "What customer emails are in home leads?",
                "How many auto leads do I have?",
                "What contact information is available in restaurant leads?"
            ],
            model_used="GPT-4o-mini (Restricted Mode)", 
            status_code="500",
            found=False,
            execution_time=round(execution_time, 2),
            filtered_by="Error occurred"
        )