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

router = APIRouter()

# Security Configuration
class SecurityConfig:
    MAX_QUERY_LENGTH = 3000
    MAX_ITERATIONS = 4
    MAX_EXECUTION_TIME = 30
    RATE_LIMIT_PER_MINUTE = 15
    
    FORBIDDEN_PATTERNS = [
        r'\b(DROP|DELETE|INSERT|UPDATE|CREATE|ALTER|TRUNCATE)\b',
        r'\b(EXEC|EXECUTE|SP_|XP_)\b',
        r'\b(LOAD_FILE|INTO\s+OUTFILE|DUMPFILE)\b',
        r'(\-\-|\#|\/\*|\*\/)',
        r'(\;.*\;)',
    ]

class AncDBInput(BaseModel):
    question: str = Field(description="Any question about the database", max_length=SecurityConfig.MAX_QUERY_LENGTH)
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

def get_intelligent_system_prompt(agent_id: Optional[str], agency_id: Optional[str]) -> str:
    """Create a focused system prompt targeting the specific *_leads tables"""
    
    filter_conditions = []
    if agent_id:
        filter_conditions.append(f"agent_id = '{agent_id}'")
    if agency_id:
        filter_conditions.append(f"agency_id = '{agency_id}'")
    
    filter_clause = " AND ".join(filter_conditions) if filter_conditions else ""
    
    return f"""You are an intelligent database assistant focused on finding customer/lead information efficiently.

SECURITY: Only use SELECT queries. For tables with agent_id/agency_id, include: WHERE {filter_clause}

PRIMARY TARGET TABLES (*_leads tables):
- auto_leads, gasstation_leads, general_business_leads, general_contractor_leads
- home_leads, hotel_leads, restaurant_leads, salon_leads, shopping_leads, tank_leads
- lead_source, lead_workflow, lead_workflow_payment, lead_workflow_reason

PRIORITY SEARCH STRATEGY:
1. FIRST PRIORITY: Search ALL *_leads tables (these contain your customer data)
2. SECOND PRIORITY: If no results, check customer_master, secondary_contact  
3. LAST RESORT: Other general tables like AgencyProfile, agent_master

EFFICIENT WORKFLOW:
1. Start with the most common *_leads tables: auto_leads, home_leads, general_business_leads
2. Check table structure: SHOW COLUMNS FROM auto_leads (or other *_leads table)
3. Search with comprehensive OR conditions across multiple person fields
4. Only check other tables if *_leads tables have no results

SMART SEARCHING PATTERN:
For "what is X's address" - search *_leads tables:
```sql
SELECT address, street, location, home_address FROM auto_leads 
WHERE {filter_clause} AND (
  name LIKE '%X%' OR 
  first_name LIKE '%X%' OR 
  email LIKE '%X%' OR 
  contact_email LIKE '%X%' OR
  phone LIKE '%X%'
) LIMIT 5
```

If no results, try next *_leads table:
```sql
SELECT address, street, location FROM home_leads 
WHERE {filter_clause} AND (
  name LIKE '%X%' OR email LIKE '%X%'
) LIMIT 5
```

INFORMATION MAPPING:
- Address questions → address, street, location, home_address, work_address
- Phone questions → phone, mobile, contact_number, work_phone, cell_phone  
- Email questions → email, contact_email, work_email

TABLE SEARCH ORDER:
1. auto_leads (largest lead table)
2. home_leads  
3. general_business_leads
4. restaurant_leads, hotel_leads, salon_leads
5. gasstation_leads, shopping_leads, tank_leads
6. If no results: customer_master, secondary_contact
7. Last resort: AgencyProfile, agent_master

EFFICIENCY RULES:
- Focus on *_leads tables first - maximum data is here
- Maximum 4-5 queries total
- Use LIMIT 5 on all queries
- Stop when you find the answer
- Try 2-3 different *_leads tables before moving to other tables

RESPONSE FORMAT:
- Found: "I found [person]'s [info]: [answer]"
- Not found: "No records found for [person] in your lead databases"

Remember: Your customer data is primarily in *_leads tables - search them thoroughly first!"""

class IntelligentAncDBAgent:
    def __init__(self, agent_id: Optional[str] = None, agency_id: Optional[str] = None):
        self.agent_id = agent_id
        self.agency_id = agency_id
        self.query_count = 0
        self.start_time = time.time()
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
            tables = self.db.get_usable_table_names()
            logger.info(f"Connected to database with {len(tables)} tables")
            return True
        except Exception as e:
            logger.error(f"Database connection failed: {e}")
            return False
    
    def initialize_agent(self) -> bool:
        try:
            self.llm = ChatOpenAI(
                model_name="gpt-4o-mini",
                temperature=0.1,
                max_tokens=2000,
                request_timeout=SecurityConfig.MAX_EXECUTION_TIME
            )
            
            toolkit = SQLDatabaseToolkit(db=self.db, llm=self.llm)
            system_message = SystemMessage(content=get_intelligent_system_prompt(self.agent_id, self.agency_id))
            
            self.agent = create_sql_agent(
                llm=self.llm,
                toolkit=toolkit,
                verbose=False,
                agent_type="openai-tools",
                system_message=system_message,
                max_iterations=SecurityConfig.MAX_ITERATIONS,
                handle_parsing_errors=True
            )
            
            logger.info(f"Agent initialized for agent_id: {self.agent_id}, agency_id: {self.agency_id}")
            return True
            
        except Exception as e:
            logger.error(f"Agent initialization failed: {e}")
            return False
    
    def process_intelligent_question(self, question: str) -> tuple:
        try:
            self.query_count += 1
            if self.query_count > SecurityConfig.RATE_LIMIT_PER_MINUTE:
                raise ValueError("Rate limit exceeded. Please wait before making more requests.")
            
            logger.info(f"Processing question: {question}")
            
            prompt = f"""
Question: "{question}"
Context: agent_id={self.agent_id}, agency_id={self.agency_id}

Search the database efficiently to answer this question. Use maximum 4 queries total.
Focus on *_leads tables first (auto_leads, home_leads, general_business_leads, etc.)

1. Quick exploration of *_leads tables
2. One targeted search query in the most relevant *_leads table
3. If needed, try one more *_leads table
4. Provide the answer

Be direct and efficient. Don't over-search.
"""
            
            response = self.agent.invoke({"input": prompt})
            result = response["output"] if isinstance(response, dict) else str(response)
            
            if "SUGGESTED_QUESTIONS:" in result:
                parts = result.split("SUGGESTED_QUESTIONS:")
                answer = parts[0].replace("ANSWER:", "").strip()
                suggestions = []
                if len(parts) > 1:
                    lines = parts[1].strip().split("\n")[:3]
                    for line in lines:
                        clean = re.sub(r'^\d+\.\s*', '', line.strip())
                        if clean:
                            suggestions.append(clean)
            else:
                answer = result.strip()
                suggestions = []
            
            default_suggestions = [
                "What other information is available for this person?",
                "Can you show me more details about this lead?",
                "How can I search for similar records?"
            ]
            while len(suggestions) < 3:
                suggestions.append(default_suggestions[len(suggestions)])
            
            return answer, suggestions[:3]
            
        except Exception as e:
            logger.error(f"Error processing question: {e}")
            return f"Error: {str(e)}", [
                "Try a simpler question",
                "Check if the data exists in your leads",
                "Contact support if needed"
            ]

# Global agent management
_intelligent_agents = {}
_agent_last_access = {}

def get_intelligent_agent(agent_id: Optional[str], agency_id: Optional[str]):
    if not agent_id and not agency_id:
        raise HTTPException(
            status_code=403,
            detail="Access denied. Valid agent_id and/or agency_id required."
        )
    
    cache_key = f"intel_{agent_id or 'none'}_{agency_id or 'none'}"
    
    current_time = time.time()
    if cache_key in _agent_last_access:
        if current_time - _agent_last_access[cache_key] < 2:
            raise HTTPException(
                status_code=429,
                detail="Please wait a moment between requests."
            )
    
    _agent_last_access[cache_key] = current_time
    
    if cache_key not in _intelligent_agents:
        try:
            _intelligent_agents[cache_key] = IntelligentAncDBAgent(agent_id=agent_id, agency_id=agency_id)
            
            if not _intelligent_agents[cache_key].test_connection():
                raise HTTPException(status_code=500, detail="Database connection failed")
            
            if not _intelligent_agents[cache_key].initialize_agent():
                raise HTTPException(status_code=500, detail="Agent initialization failed")
                
        except Exception as e:
            logger.error(f"Failed to create intelligent agent: {str(e)}")
            raise HTTPException(status_code=500, detail=f"Agent creation failed: {str(e)}")
    
    return _intelligent_agents[cache_key]

@router.post("/ask-ancdb", response_model=AncDBResponse)
async def ask_ancdb_intelligent(input_data: AncDBInput):
    """Intelligent endpoint that can handle ANY question about the ANC database"""
    start_time = time.time()
    
    try:
        intelligent_agent = get_intelligent_agent(input_data.agent_id, input_data.agency_id)
        answer, suggested_questions = intelligent_agent.process_intelligent_question(input_data.question)
        
        filter_parts = []
        if input_data.agent_id:
            filter_parts.append(f"agent_id: {input_data.agent_id}")
        if input_data.agency_id:
            filter_parts.append(f"agency_id: {input_data.agency_id}")
        filtered_by = ", ".join(filter_parts)
        
        execution_time = round(time.time() - start_time, 2)
        
        logger.info(f"Intelligent query completed in {execution_time}s")
        
        return AncDBResponse(
            question=input_data.question,
            answer=answer,
            sources=["ANC Database (Intelligent Search)"],
            suggested_questions=suggested_questions,
            model_used="GPT-4o-mini (Intelligent Mode)",
            status_code="200",
            found=True,
            execution_time=execution_time,
            filtered_by=filtered_by
        )
        
    except HTTPException:
        raise
    except Exception as e:
        execution_time = time.time() - start_time
        logger.error(f"Intelligent endpoint error: {str(e)}")
        
        return AncDBResponse(
            question=input_data.question,
            answer=f"I apologize, but I encountered an issue while searching for that information. Error: {str(e)}. Please try rephrasing your question.",
            sources=[],
            suggested_questions=[
                "Can you try asking your question in a different way?",
                "Would you like me to show you what data I can access?",
                "Can I help you with a different type of search?"
            ],
            model_used="GPT-4o-mini (Intelligent Mode)",
            status_code="500",
            found=False,
            execution_time=round(execution_time, 2),
            filtered_by="Error occurred"
        )