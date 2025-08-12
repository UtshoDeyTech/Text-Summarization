from fastapi import APIRouter, HTTPException
from typing import List
from pydantic import BaseModel, Field
from config import OPENAI_API_KEY, DB_HOST, DB_USER, DB_PASSWORD, DB_NAME, DB_PORT
from app.service.log_client import logger
from langchain_community.utilities import SQLDatabase
from langchain_openai import ChatOpenAI
from langchain_community.agent_toolkits import SQLDatabaseToolkit
from langchain_community.agent_toolkits.sql.base import create_sql_agent
from langchain.schema import SystemMessage
import time
import os

router = APIRouter()

class AncDBInput(BaseModel):
    question: str = Field(description="Question to be answered about the ANC database")

class AncDBResponse(BaseModel):
    question: str
    answer: str
    sources: List[str] = []
    suggested_questions: List[str] = []
    model_used: str
    status_code: str = "200"
    found: bool = True
    execution_time: float = 0

# System prompt for the SQL agent - matching the working test.py exactly
SYSTEM_PROMPT = """You are an expert database assistant with advanced reasoning capabilities. 

KEY INSTRUCTIONS:
- You can handle ANY question about the database, no matter how it's phrased
- Always explore the database structure first to understand what tables and columns exist
- Be creative and thorough in your search strategies
- Use LIKE, SOUNDEX, and fuzzy matching for names that might be spelled differently
- Search across ALL relevant tables and columns, not just obvious ones
- If you can't find exact matches, suggest similar or related data
- Always provide helpful context about what you found or why you couldn't find something
- For phone numbers, addresses, or contact info - check ALL tables that might contain such data
- Be persistent - try multiple search strategies if the first approach doesn't work

SEARCH STRATEGIES:
1. Start with exact matches
2. Try partial matches with LIKE '%term%'
3. Try different variations of names
4. Search in multiple related tables
5. Look for email addresses, usernames, or other identifiers
6. Check for data in unexpected places

EXAMPLE APPROACHES:
- For "What is John's phone number?" → Search users, customers, contacts, teammates, etc.
- For "Where does Sarah live?" → Look for address fields in any table with personal info
- For "rabbi" → Could be a name, username, email prefix, or nickname - search everywhere

Always be thorough and helpful, explaining your search process."""

# User prompt template for structured responses
USER_PROMPT_TEMPLATE = """Based on the database, please answer this question: {question}

After your answer, list exactly 3 suggested follow-up questions that would be valuable to ask next. 
Important: The suggested questions should be different from the current question and should help explore the data further.

Format your response as:
ANSWER: [Your clear, concise answer here]
SUGGESTED_QUESTIONS:
1. [First follow-up question - must be different from the current question]
2. [Second follow-up question - must be different from the current question]
3. [Third follow-up question - must be different from the current question]"""

class AncDBAgent:
    def __init__(self):
        """Initialize the ANC Database Agent"""
        self.setup_environment()
        self.connection_url = self.build_connection_url()
        self.db = None
        self.agent = None
        
    def setup_environment(self):
        """Set up environment variables"""
        if not OPENAI_API_KEY:
            raise ValueError("OPENAI_API_KEY not found in environment variables")
        
        os.environ['OPENAI_API_KEY'] = OPENAI_API_KEY
        
    def build_connection_url(self) -> str:
        """Build database connection URL from config"""
        if all([DB_HOST, DB_USER, DB_PASSWORD, DB_NAME, DB_PORT]):
            logger.info(f"Using configured database: {DB_HOST}")
            return f"mysql+pymysql://{DB_USER}:{DB_PASSWORD}@{DB_HOST}:{DB_PORT}/{DB_NAME}"
        else:
            logger.warning("Database config incomplete, using fallback connection")
            return "mysql+pymysql://mypolicylistdb:Anc!2024@174.138.44.203:3306/mypolicylistdb"
    
    def test_database_connection(self) -> bool:
        """Test database connection and discover structure"""
        try:
            self.db = SQLDatabase.from_uri(self.connection_url)
            tables = self.db.get_usable_table_names()
            logger.info(f"Database connected successfully with {len(tables)} tables")
            
            # Auto-discover key information about the database
            self.discover_database_structure()
            return True
        except Exception as e:
            logger.error(f"Database connection failed: {e}")
            return False
    
    def discover_database_structure(self):
        """Automatically discover and understand the database structure"""
        try:
            tables = self.db.get_usable_table_names()
            
            # Find tables that likely contain user/customer information
            user_tables = []
            for table in tables:
                table_lower = table.lower()
                if any(keyword in table_lower for keyword in ['user', 'customer', 'contact', 'person', 'member', 'client']):
                    user_tables.append(table)
            
            if user_tables:
                logger.info(f"Found user-related tables: {user_tables}")
            
            # Sample some data to understand the content
            sample_info = []
            for table in tables[:5]:  # Check first 5 tables
                try:
                    sample = self.db.run(f"SELECT * FROM {table} LIMIT 1")
                    if sample and sample != '[]':
                        sample_info.append(f"{table}: Has data")
                except:
                    pass
            
            if sample_info:
                logger.info(f"Tables with data: {len(sample_info)}")
                
        except Exception as e:
            logger.warning(f"Could not fully discover database structure: {e}")
    
    def initialize_agent(self) -> bool:
        """Initialize the SQL agent"""
        try:
            # Use GPT-4o-mini for better reasoning
            self.llm = ChatOpenAI(
                model_name="gpt-4o-mini",
                temperature=0,
                max_tokens=2000
            )
            logger.info("LLM initialized with enhanced model")
            
            # Create toolkit
            toolkit = SQLDatabaseToolkit(db=self.db, llm=self.llm)
            
            # Create system message
            system_message = SystemMessage(content=SYSTEM_PROMPT)
            
            self.agent = create_sql_agent(
                llm=self.llm,
                toolkit=toolkit,
                verbose=True,
                agent_type="openai-tools",
                system_message=system_message,
                max_iterations=10,
                early_stopping_method="generate"
            )
            logger.info("SQL Agent initialized successfully")
            return True
            
        except Exception as e:
            logger.error(f"Agent initialization failed: {e}")
            return False
    
    def ask_question(self, question: str) -> tuple:
        """Process question and return answer with suggested questions"""
        try:
            logger.info(f"Processing question: {question}")
            
            # Use the same format as the working test.py
            formatted_question = USER_PROMPT_TEMPLATE.format(question=question)
            
            response = self.agent.invoke({"input": formatted_question})
            result = response["output"] if isinstance(response, dict) else str(response)
            
            # Parse the response to extract answer and suggested questions
            answer_parts = result.split("SUGGESTED_QUESTIONS:")
            main_answer = answer_parts[0].replace("ANSWER:", "").strip()
            suggested_questions = []
            
            if len(answer_parts) > 1:
                questions_text = answer_parts[1].strip()
                for line in questions_text.split("\n"):
                    if line.strip() and any(line.strip().startswith(str(i)) for i in range(1, 4)):
                        question_text = line.strip().split(".", 1)[1].strip() if "." in line else line.strip()
                        suggested_questions.append(question_text)
            
            return main_answer, suggested_questions
            
        except Exception as e:
            logger.error(f"Error processing question: {e}")
            return f"I encountered an error: {e}. Please try rephrasing your question.", []

# Global agent instance
_agent_instance = None

def get_agent():
    """Get or create the global agent instance"""
    global _agent_instance
    if _agent_instance is None:
        _agent_instance = AncDBAgent()
        if not _agent_instance.test_database_connection():
            raise HTTPException(
                status_code=500,
                detail="Database connection failed"
            )
        if not _agent_instance.initialize_agent():
            raise HTTPException(
                status_code=500,
                detail="Agent initialization failed"
            )
    return _agent_instance

@router.post("/ask-ancdb", response_model=AncDBResponse)
async def ask_ancdb(input_data: AncDBInput):
    """Ask questions about the ANC database using natural language"""
    start_time = time.time()
    
    try:
        # Get or initialize the agent
        agent = get_agent()
        
        # Process the question
        answer, suggested_questions = agent.ask_question(input_data.question)
        
        # Return successful response
        return AncDBResponse(
            question=input_data.question,
            answer=answer,
            sources=["ANC Database"],
            suggested_questions=suggested_questions,
            model_used="GPT-4o-mini",
            status_code="200",
            found=True,
            execution_time=round(time.time() - start_time, 2)
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error in ask_ancdb: {str(e)}")
        execution_time = time.time() - start_time
        
        return AncDBResponse(
            question=input_data.question,
            answer=f"I encountered an error while processing your question: {str(e)}",
            sources=[],
            suggested_questions=[],
            model_used="GPT-4o-mini",
            status_code="500",
            found=False,
            execution_time=round(execution_time, 2)
        )