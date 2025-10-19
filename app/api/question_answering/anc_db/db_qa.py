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
from langchain.callbacks.base import BaseCallbackHandler

router = APIRouter()

# Security Configuration
class SecurityConfig:
    MAX_QUERY_LENGTH = 3000
    MAX_ITERATIONS = 10  # Increased to allow for schema exploration + JOIN operations
    MAX_EXECUTION_TIME = 60
    RATE_LIMIT_PER_MINUTE = 15
    
    FORBIDDEN_PATTERNS = [
        r'\b(DROP|DELETE|INSERT|UPDATE|CREATE|ALTER|TRUNCATE)\b',
        r'\b(EXEC|EXECUTE|SP_|XP_)\b',
        r'\b(LOAD_FILE|INTO\s+OUTFILE|DUMPFILE)\b',
        r'(\-\-|\#|\/\*|\*\/)',
        r'(\;.*\;)',
    ]

# Lead type to table mapping
ALLOWED_TABLES = {
    "Home": "home_leads", 
    "Auto": "auto_leads", 
    "Gas Station": "gasstation_leads", 
    "Restaurant": "restaurant_leads", 
    "Salon": "salon_leads", 
    "General Contractor": "general_contractor_leads", 
    "Shopping Mall": "shopping_leads", 
    "General Business": "general_business_leads", 
    "Hotel/Motel": "hotel_leads",
    "Tank": "tank_leads"
}

class IterationLoggingCallback(BaseCallbackHandler):
    """Custom callback to log each iteration of the agent"""
    
    def __init__(self):
        self.iteration_count = 0
        self.actions_taken = []
        self.observations = []
        
    def on_agent_action(self, action: AgentAction, **kwargs) -> None:
        """Log when agent takes an action"""
        self.iteration_count += 1
        self.actions_taken.append({
            'iteration': self.iteration_count,
            'tool': action.tool,
            'tool_input': action.tool_input,
            'log': action.log[:200] if action.log else None
        })
        logger.info(f"[ITERATION {self.iteration_count}] Agent action: {action.tool}")
        logger.debug(f"[ITERATION {self.iteration_count}] Tool input: {action.tool_input}")
        
    def on_tool_end(self, output: str, **kwargs) -> None:
        """Log when a tool completes"""
        self.observations.append({
            'iteration': self.iteration_count,
            'output': output[:200] if output else None
        })
        logger.debug(f"[ITERATION {self.iteration_count}] Tool output: {output[:200]}...")
        
    def on_agent_finish(self, finish: AgentFinish, **kwargs) -> None:
        """Log when agent finishes"""
        logger.info(f"[AGENT FINISH] Total iterations: {self.iteration_count}")
        logger.debug(f"[AGENT FINISH] Final output: {finish.return_values}")
        
    def on_llm_start(self, serialized: dict, prompts: List[str], **kwargs) -> None:
        """Log when LLM starts processing"""
        logger.debug(f"[ITERATION {self.iteration_count}] LLM started processing")
        
    def on_llm_end(self, response, **kwargs) -> None:
        """Log when LLM completes"""
        logger.debug(f"[ITERATION {self.iteration_count}] LLM completed processing")

class AncDBInput(BaseModel):
    question: str = Field(description="Any question about leads or customer data", max_length=SecurityConfig.MAX_QUERY_LENGTH)
    lead_type: str = Field(description="Type of lead to focus on (Home, Auto, Gas Station, etc.)", max_length=100)
    agent_id: str = Field(description="Agent ID for filtering (required for security)", max_length=100)
    agency_id: str = Field(description="Agency ID for filtering (required for security)", max_length=100)

    @validator('lead_type')
    def validate_lead_type(cls, v):
        if not v or not v.strip():
            logger.warning("[VALIDATION] Empty lead_type provided")
            raise ValueError("Lead type cannot be empty")
        
        v = v.strip()
        if v not in ALLOWED_TABLES:
            logger.warning(f"[VALIDATION] Invalid lead_type: {v}")
            valid_types = ", ".join(ALLOWED_TABLES.keys())
            raise ValueError(f"Invalid lead type. Must be one of: {valid_types}")
        
        logger.info(f"[VALIDATION] Lead type validated: {v} -> {ALLOWED_TABLES[v]}")
        return v

    @validator('question')
    def validate_question(cls, v):
        if not v or not v.strip():
            logger.warning("[VALIDATION] Empty question provided")
            raise ValueError("Question cannot be empty")
        
        logger.debug(f"[VALIDATION] Validating question: {v[:100]}...")
        
        for pattern in SecurityConfig.FORBIDDEN_PATTERNS:
            if re.search(pattern, v, re.IGNORECASE):
                logger.warning(f"[SECURITY] Suspicious pattern detected in question: {pattern}")
        
        logger.info("[VALIDATION] Question validation passed")
        return v.strip()

    @validator('agent_id', 'agency_id')
    def validate_ids(cls, v):
        if not v or not v.strip():
            logger.warning(f"[VALIDATION] Empty ID provided")
            raise ValueError("agent_id and agency_id are required for security")
        
        if not re.match(r'^[a-zA-Z0-9_-]+$', v):
            logger.warning(f"[VALIDATION] Invalid ID format: {v}")
            raise ValueError("ID contains invalid characters")
        logger.debug(f"[VALIDATION] ID validated: {v}")
        return v.strip()

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
    lead_type: str = ""

def format_response_as_html_list(text: str) -> str:
    """
    Convert text responses containing multiple items into HTML list format.
    Detects numbered lists, bullet points, and multiple distinct items.
    """
    if not text or len(text.strip()) < 10:
        return text
    
    logger.debug(f"[FORMAT] Attempting to format response as HTML list")
    
    # Clean the text
    text = text.strip()
    
    # Pattern 1: Detect numbered lists (1. item, 2. item, etc.)
    numbered_pattern = re.compile(r'^(\d+\.\s+.+?)(?=\n\d+\.|$)', re.MULTILINE | re.DOTALL)
    numbered_matches = numbered_pattern.findall(text)
    
    if len(numbered_matches) >= 2:
        logger.debug(f"[FORMAT] Detected numbered list with {len(numbered_matches)} items")
        items = []
        for match in numbered_matches:
            item = re.sub(r'^\d+\.\s*', '', match.strip())
            if item:
                items.append(f"<li>{item}</li>")
        
        if items:
            logger.info(f"[FORMAT] Formatted as ordered list with {len(items)} items")
            return f"<ol>{''.join(items)}</ol>"
    
    # Pattern 2: Detect bullet points (-, *, •, etc.)
    bullet_pattern = re.compile(r'^([-*•]\s+.+?)(?=\n[-*•]|$)', re.MULTILINE | re.DOTALL)
    bullet_matches = bullet_pattern.findall(text)
    
    if len(bullet_matches) >= 2:
        logger.debug(f"[FORMAT] Detected bullet list with {len(bullet_matches)} items")
        items = []
        for match in bullet_matches:
            item = re.sub(r'^[-*•]\s*', '', match.strip())
            if item:
                items.append(f"<li>{item}</li>")
        
        if items:
            logger.info(f"[FORMAT] Formatted as unordered list with {len(items)} items")
            return f"<ul>{''.join(items)}</ul>"
    
    # Pattern 3: Detect multiple distinct sentences/paragraphs that could be list items
    lines = [line.strip() for line in text.split('\n') if line.strip()]
    
    if len(lines) >= 2 and len(lines) <= 10:
        logger.debug(f"[FORMAT] Checking {len(lines)} lines for structured data")
        email_count = sum(1 for line in lines if re.search(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b', line))
        phone_count = sum(1 for line in lines if re.search(r'\b\d{3}[-.]?\d{3}[-.]?\d{4}\b', line))
        name_count = sum(1 for line in lines if re.search(r'^[A-Z][a-z]+ [A-Z][a-z]+', line))
        
        if (email_count >= len(lines) * 0.6 or 
            phone_count >= len(lines) * 0.6 or 
            name_count >= len(lines) * 0.6 or
            all(len(line) > 10 and len(line) < 200 for line in lines)):
            
            logger.info(f"[FORMAT] Formatted structured data as unordered list")
            items = [f"<li>{line}</li>" for line in lines]
            return f"<ul>{''.join(items)}</ul>"
    
    # Pattern 4: Detect comma-separated lists that should be converted
    if ', ' in text and text.count(',') >= 2:
        parts = [part.strip() for part in text.split(',')]
        if len(parts) >= 3 and len(parts) <= 15:
            avg_length = sum(len(part) for part in parts) / len(parts)
            if 5 <= avg_length <= 50 and not any('\n' in part for part in parts):
                logger.info(f"[FORMAT] Formatted comma-separated list with {len(parts)} items")
                items = [f"<li>{part}</li>" for part in parts if part]
                return f"<ul>{''.join(items)}</ul>"
    
    logger.debug(f"[FORMAT] No formatting pattern matched, returning original text")
    return text

def get_restricted_system_prompt(lead_type: str, table_name: str, agent_id: str, agency_id: str, table_schema: str) -> str:
    """Create a system prompt that allows JOIN with customer_master table"""
    
    logger.debug(f"[PROMPT] Generating system prompt for lead_type={lead_type}, table={table_name}")
    
    filter_clause = f"WHERE {table_name}.agent_id = '{agent_id}' AND {table_name}.agency_id = '{agency_id}'"
    
    return f"""You are a specialized database assistant for {lead_type} leads data with access to customer information.

AVAILABLE TABLES:
1. {table_name} - Contains lead-specific data with {table_schema.count('|')} columns
2. customer_master - Contains customer information (customer_id, agency_id, agent_id, first_name, last_name, email_id, mobile_number, work_number, created_date, updated_date)

TABLE SCHEMA FOR {table_name}:
{table_schema}

RELATIONSHIP:
- {table_name}.customer_id = customer_master.customer_id (Foreign Key relationship)
- ALWAYS use this JOIN when customer names or customer_master data is needed

MANDATORY FILTERING (SECURITY):
- EVERY query on {table_name} MUST include: {filter_clause}
- This filtering is REQUIRED for security - never skip it
- When JOINing with customer_master, apply the filter on {table_name}

CRITICAL INSTRUCTIONS FOR HANDLING LARGE TABLE:
1. ALWAYS use the sql_db_schema tool FIRST to understand available columns before querying
2. When user asks about ANY field, check the schema to find the exact column name
3. Use sql_db_schema tool to explore table structure when uncertain about column names
4. The table has 70-100 columns, so explore the schema to find relevant columns
5. Look for columns that match the user's question semantically (e.g., "premium" might be in "annual_premium", "monthly_premium", etc.)

QUERY PATTERNS:

1. For customer name searches (e.g., "find email of John", "show data for Sarah"):
   ```sql
   SELECT cm.first_name, cm.last_name, l.email, l.contact_email, l.mobile_number
   FROM {table_name} l
   INNER JOIN customer_master cm ON l.customer_id = cm.customer_id
   {filter_clause}
   AND (cm.first_name LIKE '%name%' OR cm.last_name LIKE '%name%')
   LIMIT 10
   ```

2. For general data queries without name search:
   ```sql
   SELECT email, contact_email, mobile_number
   FROM {table_name}
   {filter_clause}
   LIMIT 10
   ```

3. For combined customer and lead information:
   ```sql
   SELECT cm.first_name, cm.last_name, cm.email_id as customer_email,
          l.email as lead_email, l.mobile_number, l.location_address
   FROM {table_name} l
   INNER JOIN customer_master cm ON l.customer_id = cm.customer_id
   {filter_clause}
   LIMIT 10
   ```

CRITICAL RULES:
- Use INNER JOIN when you need customer names or customer_master data
- ALWAYS include the security filter: {filter_clause}
- Use LIKE '%term%' for flexible text searching on names
- Check BOTH cm.first_name and cm.last_name when searching by name
- Use LIMIT to avoid returning too many results
- NEVER mention table names in responses - only say "{lead_type} leads" or "customers"

DATA LOCATION GUIDE:
- Customer names (first_name, last_name): customer_master table
- Emails: Can be in both tables (email_id in customer_master, email/contact_email in {table_name})
- Phone numbers: Can be in both tables (mobile_number, work_number in both)
- Addresses: Primarily in {table_name} (location_address, mailing_address)
- Dates: created_date, updated_date in both tables

RESPONSE GUIDELINES:
- ALWAYS check the table schema first before writing queries
- If uncertain about column names, use sql_db_schema to explore
- Search for columns semantically related to user's question
- If answer exists: Provide the specific data found with customer names when relevant
- If column not found in schema: Tell user that specific data field is not available in {lead_type} leads
- If no data found: Generate a friendly message saying no information was found
- If query is unclear: Ask for clarification
- NEVER expose table names or technical details to the user
- Always explain what you searched for in natural language
- Format multiple results clearly (one per line or in a structured format)

WORKFLOW FOR EVERY QUERY:
1. Analyze the user's question to identify what data they want
2. Use sql_db_schema tool to check available columns in {table_name}
3. Find matching columns (exact match or semantic match)
4. Construct appropriate SELECT query with proper JOIN if needed
5. Apply mandatory security filters
6. Execute query and format results

SECURITY REMINDERS:
- NEVER query tables other than {table_name} and customer_master
- ALWAYS apply security filters on {table_name}
- agent_id = '{agent_id}' AND agency_id = '{agency_id}' must be in WHERE clause for {table_name}
- Only use SELECT queries - no modifications allowed"""

class RestrictedAncDBAgent:
    def __init__(self, lead_type: str, agent_id: str, agency_id: str):
        logger.info(f"[AGENT INIT] Initializing RestrictedAncDBAgent for lead_type={lead_type}")
        
        if lead_type not in ALLOWED_TABLES:
            raise ValueError(f"Invalid lead_type: {lead_type}")
        
        if not agent_id or not agent_id.strip():
            raise ValueError("agent_id is required for security")
        
        if not agency_id or not agency_id.strip():
            raise ValueError("agency_id is required for security")
        
        self.lead_type = lead_type
        self.table_name = ALLOWED_TABLES[lead_type]
        self.agent_id = agent_id.strip()
        self.agency_id = agency_id.strip()
        
        logger.info(f"[AGENT INIT] Tables allowed: {self.table_name}, customer_master")
        logger.info(f"[AGENT INIT] Security filters: agent_id={self.agent_id}, agency_id={self.agency_id}")
        
        self.setup_environment()
        self.connection_url = self.build_connection_url()
        self.db = None
        self.agent = None
        logger.info(f"[AGENT INIT] Agent instance created successfully")
        
    def setup_environment(self):
        logger.debug("[ENV SETUP] Setting up environment variables")
        if not OPENAI_API_KEY:
            logger.error("[ENV SETUP] OPENAI_API_KEY not found in environment variables")
            raise ValueError("OPENAI_API_KEY not found in environment variables")
        os.environ['OPENAI_API_KEY'] = OPENAI_API_KEY
        logger.info("[ENV SETUP] Environment setup completed successfully")
        
    def build_connection_url(self) -> str:
        """Build database connection URL from environment variables"""
        logger.debug("[DB CONFIG] Building database connection URL")
        
        if all([DB_HOST, DB_USER, DB_PASSWORD, DB_NAME, DB_PORT]):
            logger.info(f"[DB CONFIG] Using configured database: {DB_HOST}:{DB_PORT}/{DB_NAME}")
            encoded_password = quote(DB_PASSWORD)
            connection_url = f"mysql+pymysql://{DB_USER}:{encoded_password}@{DB_HOST}:{DB_PORT}/{DB_NAME}?charset=utf8mb4"
            logger.info("[DB CONFIG] Connection URL built successfully")
            return connection_url
        else:
            missing_vars = []
            if not DB_HOST: missing_vars.append("DB_HOST")
            if not DB_USER: missing_vars.append("DB_USER")
            if not DB_PASSWORD: missing_vars.append("DB_PASSWORD")
            if not DB_NAME: missing_vars.append("DB_NAME")
            if not DB_PORT: missing_vars.append("DB_PORT")
            
            logger.error(f"[DB CONFIG] Database configuration is incomplete. Missing: {', '.join(missing_vars)}")
            raise ValueError("Database configuration error")
    
    def test_connection(self) -> bool:
        """Test database connection and verify tables exist"""
        logger.info("[DB CONNECTION] Testing database connection...")
        try:
            start_time = time.time()
            # Include both the lead table and customer_master
            self.db = SQLDatabase.from_uri(
                self.connection_url,
                include_tables=[self.table_name, "customer_master"],
                sample_rows_in_table_info=2  # Show sample data in schema
            )
            connection_time = round(time.time() - start_time, 3)
            
            logger.info(f"[DB CONNECTION] ✓ Database connected successfully in {connection_time}s")
            
            # Verify tables are accessible
            usable_tables = self.db.get_usable_table_names()
            
            if self.table_name in usable_tables and "customer_master" in usable_tables:
                logger.info(f"[DB CONNECTION] ✓ Tables accessible: {self.table_name}, customer_master")
                
                # Get and log table schema for the leads table
                try:
                    table_info = self.db.get_table_info_no_throw([self.table_name])
                    column_count = table_info.count('\n')
                    logger.info(f"[DB CONNECTION] ✓ {self.table_name} has approximately {column_count} columns")
                    logger.debug(f"[DB CONNECTION] Schema preview: {table_info[:500]}...")
                except Exception as schema_error:
                    logger.warning(f"[DB CONNECTION] Could not retrieve schema info: {schema_error}")
                    
            else:
                missing = []
                if self.table_name not in usable_tables:
                    missing.append(self.table_name)
                if "customer_master" not in usable_tables:
                    missing.append("customer_master")
                logger.error(f"[DB CONNECTION] ✗ Tables not found: {', '.join(missing)}")
                return False
            
            logger.info(f"[DB CONNECTION] ✓ Connection test passed")
            return True
            
        except Exception as e:
            logger.error(f"[DB CONNECTION] ✗ Database connection failed: {type(e).__name__}: {str(e)}")
            return False
    
    def get_table_schema_summary(self) -> str:
        """Get a formatted summary of the table schema"""
        try:
            table_info = self.db.get_table_info_no_throw([self.table_name])
            
            # Parse column names from the schema
            columns = []
            for line in table_info.split('\n'):
                # Extract column names (format varies, but typically starts with column name)
                if line.strip() and not line.strip().startswith('CREATE') and not line.strip().startswith('/*'):
                    # Try to extract column name (before space, comma, or parenthesis)
                    match = re.match(r'^\s*`?(\w+)`?\s+', line)
                    if match:
                        columns.append(match.group(1))
            
            # Format as readable list
            if columns:
                schema_summary = "Available columns:\n" + "\n".join([f"  - {col}" for col in columns[:50]])  # Limit to first 50 for readability
                if len(columns) > 50:
                    schema_summary += f"\n  ... and {len(columns) - 50} more columns"
                return schema_summary
            else:
                return table_info[:1000]  # Fallback to raw schema info
                
        except Exception as e:
            logger.error(f"[SCHEMA] Error getting schema summary: {e}")
            return "Schema information not available"
    
    def initialize_agent(self) -> bool:
        logger.info("[AGENT INIT] Initializing SQL agent...")
        try:
            start_time = time.time()
            
            logger.debug("[AGENT INIT] Creating ChatOpenAI LLM instance")
            self.llm = ChatOpenAI(
                model_name="gpt-4o-mini",
                temperature=0,
                max_tokens=2000,
                request_timeout=SecurityConfig.MAX_EXECUTION_TIME
            )
            logger.info(f"[AGENT INIT] ✓ LLM initialized (model: gpt-4o-mini)")
            
            logger.debug("[AGENT INIT] Creating SQL Database Toolkit")
            toolkit = SQLDatabaseToolkit(db=self.db, llm=self.llm)
            logger.info("[AGENT INIT] ✓ Toolkit created")
            
            # Get table schema summary
            logger.debug("[AGENT INIT] Retrieving table schema")
            table_schema = self.get_table_schema_summary()
            logger.info(f"[AGENT INIT] ✓ Schema retrieved for {self.table_name}")
            
            logger.debug("[AGENT INIT] Generating system message")
            system_message = SystemMessage(
                content=get_restricted_system_prompt(
                    self.lead_type, 
                    self.table_name, 
                    self.agent_id, 
                    self.agency_id,
                    table_schema
                )
            )
            logger.info("[AGENT INIT] ✓ System message generated")
            
            logger.debug(f"[AGENT INIT] Creating agent with max_iterations={SecurityConfig.MAX_ITERATIONS}")
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
            
            init_time = round(time.time() - start_time, 3)
            logger.info(f"[AGENT INIT] ✓ Agent initialized successfully in {init_time}s")
            return True
            
        except Exception as e:
            logger.error(f"[AGENT INIT] ✗ Agent initialization failed: {type(e).__name__}: {str(e)}")
            return False
    
    def generate_no_results_message(self, question: str) -> str:
        """Generate a natural, AI-created message when no results are found"""
        try:
            logger.debug("[NO RESULTS] Generating AI-powered 'no results' message")
            
            no_results_llm = ChatOpenAI(
                model_name="gpt-4o-mini",
                temperature=0.8,
                max_tokens=150
            )

            prompt = f"""The user asked: "{question}" about {self.lead_type} leads.

After searching the database (including customer information), no relevant information was found.

Generate a friendly, natural, conversational response (2-3 sentences) that:
1. Acknowledges their question
2. States that no relevant information was found in {self.lead_type} leads
3. Encourages them to try a different search or be more specific
4. Uses varied, natural language (avoid templates)
5. NEVER mention table names

Keep it concise, helpful, and professional."""

            response = no_results_llm.invoke(prompt)
            message = response.content if hasattr(response, 'content') else str(response)
            
            logger.info(f"[NO RESULTS] Generated message: {message[:100]}...")
            return message.strip()

        except Exception as e:
            logger.error(f"[NO RESULTS] Error generating message: {e}")
            return f"I couldn't find any relevant information in your {self.lead_type} leads for that query. Please try rephrasing your question or providing more specific details."
    
    def generate_natural_suggestions(self, question: str, found_data: bool) -> List[str]:
        """Generate natural, contextual suggested questions using AI"""
        try:
            logger.debug("[SUGGESTIONS] Generating AI-powered suggestions")
            
            suggestions_llm = ChatOpenAI(
                model_name="gpt-4o-mini",
                temperature=0.7,
                max_tokens=200
            )

            data_context = "data was found" if found_data else "no data was found"
            
            prompt = f"""The user asked: "{question}" about {self.lead_type} leads, and {data_context}.

Generate exactly 3 helpful follow-up questions they might want to ask about {self.lead_type} leads.

Requirements:
- Each question should be natural and conversational
- Focus on {self.lead_type} leads and customer information
- Include questions about customer names, emails, phone numbers, and addresses
- NEVER mention table names
- Only say "{self.lead_type} leads" or "customers"
- Keep each question under 15 words
- Make them practical and useful

Format: Just list 3 questions, one per line, no numbering."""

            response = suggestions_llm.invoke(prompt)
            suggestions_text = response.content if hasattr(response, 'content') else str(response)
            
            # Parse suggestions
            suggestions = [s.strip() for s in suggestions_text.strip().split('\n') if s.strip()]
            suggestions = [re.sub(r'^\d+[\.\)]\s*', '', s) for s in suggestions]
            suggestions = suggestions[:3]
            
            # Fallback if we don't get 3
            while len(suggestions) < 3:
                fallback = [
                    f"Show me customer names and emails from {self.lead_type} leads",
                    f"How many customers do I have in {self.lead_type} leads?",
                    f"Find contact information for a specific customer name"
                ]
                suggestions.append(fallback[len(suggestions)])
            
            logger.info(f"[SUGGESTIONS] Generated {len(suggestions)} suggestions")
            return suggestions[:3]

        except Exception as e:
            logger.error(f"[SUGGESTIONS] Error generating suggestions: {e}")
            return [
                f"Show me customer names from {self.lead_type} leads",
                f"Find email addresses with customer names",
                f"How many customers are in my {self.lead_type} leads?"
            ]
    
    def process_question(self, question: str) -> tuple:
        logger.info(f"[PROCESS] Processing question: {question[:100]}...")
        logger.info(f"[PROCESS] Lead type: {self.lead_type}, Table: {self.table_name}")
        logger.info(f"[PROCESS] Security filters: agent_id={self.agent_id}, agency_id={self.agency_id}")
        
        try:
            filter_clause = f"{self.table_name}.agent_id = '{self.agent_id}' AND {self.table_name}.agency_id = '{self.agency_id}'"
            
            prompt = f"""
Question: {question}

AVAILABLE TABLES:
- {self.table_name}: Lead-specific data (has 70-100+ columns - use sql_db_schema to explore)
- customer_master: Customer information (names, contact details)
- JOIN ON: {self.table_name}.customer_id = customer_master.customer_id

CRITICAL SECURITY REQUIREMENT:
- MUST filter {self.table_name} by: WHERE {filter_clause}
- These filters are MANDATORY for every query on {self.table_name}

IMPORTANT WORKFLOW - FOLLOW THESE STEPS:
1. FIRST: Use sql_db_schema tool to check what columns exist in {self.table_name}
2. SECOND: Identify which columns match the user's question (look for semantic matches)
3. THIRD: Determine if you need customer names (if yes, JOIN with customer_master)
4. FOURTH: Construct your SELECT query with the correct columns
5. FIFTH: Apply mandatory security filters
6. SIXTH: Execute and format results

COLUMN SEARCH STRATEGY:
- User asks about "premium" → look for columns like: premium, annual_premium, monthly_premium, total_premium
- User asks about "policy" → look for: policy_number, policy_type, policy_status, policy_start_date
- User asks about "address" → look for: address, location_address, mailing_address, street_address
- User asks about "phone" → look for: phone, mobile_number, work_number, contact_number, phone_number
- User asks about "status" → look for: status, lead_status, policy_status, account_status
- Be flexible and search semantically, not just exact matches

QUERY CONSTRUCTION:
- If question mentions customer NAME: Use INNER JOIN with customer_master
  Example for "What is the premium for customer John?":
  ```
  SELECT cm.first_name, cm.last_name, l.annual_premium, l.monthly_premium
  FROM {self.table_name} l
  INNER JOIN customer_master cm ON l.customer_id = cm.customer_id
  WHERE {filter_clause}
  AND (cm.first_name LIKE '%John%' OR cm.last_name LIKE '%John%')
  LIMIT 10
  ```

- If question asks about lead data WITHOUT customer name:
  ```
  SELECT column1, column2, column3
  FROM {self.table_name}
  WHERE {filter_clause}
  LIMIT 10
  ```

- If question asks to list customers WITH their lead data:
  ```
  SELECT cm.first_name, cm.last_name, l.column1, l.column2
  FROM {self.table_name} l
  INNER JOIN customer_master cm ON l.customer_id = cm.customer_id
  WHERE {filter_clause}
  LIMIT 10
  ```

RESPONSE RULES:
- If column doesn't exist in schema: Politely tell user that specific field is not available
- If data found: Show it with customer names when relevant
- If no data found: Say no information found for this query
- If unclear: Ask for clarification
- Format multiple results clearly (one per line)
- NEVER mention table names in your response to the user

Format your final response as:
ANSWER: [Your specific answer with data]
SUGGESTED_QUESTIONS:
1. [Related question about available data]
2. [Another helpful question]
3. [Third useful question]
"""
            
            iteration_callback = IterationLoggingCallback()
            
            logger.info("[AGENT INVOKE] Starting agent execution...")
            start_time = time.time()
            
            try:
                response = self.agent.invoke(
                    {"input": prompt},
                    config={"callbacks": [iteration_callback]}
                )
                
                execution_time = round(time.time() - start_time, 3)
                result = response["output"] if isinstance(response, dict) else str(response)
                
                logger.info(f"[AGENT INVOKE] ✓ Agent completed in {execution_time}s")
                logger.info(f"[AGENT INVOKE] Iterations: {iteration_callback.iteration_count}/{SecurityConfig.MAX_ITERATIONS}")
                
            except Exception as agent_error:
                execution_time = round(time.time() - start_time, 3)
                logger.error(f"[AGENT INVOKE] Error after {execution_time}s: {str(agent_error)[:500]}")
                raise agent_error
            
            # Parse the response
            logger.debug("[PROCESS] Parsing agent response")
            if "SUGGESTED_QUESTIONS:" in result:
                parts = result.split("SUGGESTED_QUESTIONS:")
                answer = parts[0].replace("ANSWER:", "").strip()
                suggestions = []
                if len(parts) > 1:
                    lines = parts[1].strip().split("\n")
                    for line in lines:
                        clean = re.sub(r'^\d+[\.\)]\s*', '', line.strip())
                        if clean and len(suggestions) < 3:
                            suggestions.append(clean)
            else:
                answer = result.strip()
                suggestions = []
            
            # Check if no data was found or column doesn't exist
            not_found_indicators = [
                "no relevant", "not found", "couldn't find", "no information",
                "not available", "doesn't exist", "no such column", "no data"
            ]
            
            if not answer or len(answer) < 10 or any(indicator in answer.lower() for indicator in not_found_indicators):
                logger.warning(f"[PROCESS] No relevant data found or column unavailable")
                
                # Check if it's a "column not found" vs "no data" scenario
                if any(term in answer.lower() for term in ["not available", "doesn't exist", "no such column"]):
                    # Column doesn't exist - keep the agent's explanation
                    suggestions = [
                        f"What types of information are available in {self.lead_type} leads?",
                        f"Show me customer names and contact details",
                        f"List some sample data from {self.lead_type} leads"
                    ]
                else:
                    # No data found - generate friendly message
                    answer = self.generate_no_results_message(question)
                    suggestions = self.generate_natural_suggestions(question, found_data=False)
            else:
                suggestions = self.generate_natural_suggestions(question, found_data=True)
            
            formatted_answer = format_response_as_html_list(answer)
            
            logger.info(f"[PROCESS] ✓ Question processed successfully")
            return formatted_answer, suggestions[:3]
            
        except Exception as e:
            logger.error(f"[PROCESS] ✗ Error: {type(e).__name__}: {str(e)}")
            raise

# Global agent management
_restricted_agents = {}

def get_restricted_agent(lead_type: str, agent_id: str, agency_id: str):
    """Get or create a restricted agent instance for specific lead type"""
    cache_key = f"restricted_{lead_type}_{agent_id}_{agency_id}"
    
    if cache_key not in _restricted_agents:
        try:
            logger.info(f"[CACHE] Creating new agent instance: {cache_key}")
            _restricted_agents[cache_key] = RestrictedAncDBAgent(
                lead_type=lead_type,
                agent_id=agent_id, 
                agency_id=agency_id
            )
            
            if not _restricted_agents[cache_key].test_connection():
                logger.error(f"[CACHE] Database connection failed for {cache_key}")
                raise HTTPException(status_code=500, detail="Database connection failed")
            
            if not _restricted_agents[cache_key].initialize_agent():
                logger.error(f"[CACHE] Agent initialization failed for {cache_key}")
                raise HTTPException(status_code=500, detail="Agent initialization failed")
            
            logger.info(f"[CACHE] Agent instance created successfully: {cache_key}")
                
        except HTTPException:
            raise
        except Exception as e:
            logger.error(f"[CACHE] Failed to create restricted agent: {str(e)}")
            raise HTTPException(status_code=500, detail=f"Agent creation failed: {str(e)}")
    else:
        logger.info(f"[CACHE] Using existing agent instance: {cache_key}")
    
    return _restricted_agents[cache_key]

@router.post("/ask-ancdb", response_model=AncDBResponse)
async def ask_ancdb_restricted(input_data: AncDBInput):
    """
    Ask questions about specific lead type data with mandatory security filtering.
    Now supports queries about customer names by joining with customer_master table.
    
    All 4 fields are required:
    - question: Your query about the leads or customers (e.g., "what is the email of John?")
    - lead_type: Type of lead (Home, Auto, Restaurant, etc.)
    - agent_id: Your agent ID (required for security)
    - agency_id: Your agency ID (required for security)
    
    Examples:
    - "What is the email address of Utsho?"
    - "Show me all customer names and their emails"
    - "Find phone number for Sarah"
    - "List customers with their contact information"
    
    all of these example will be like an Insurance agent is asking some question about his leads.
    """
    start_time = time.time()
    
    try:
        logger.info(f"[ENDPOINT] Received request for lead_type={input_data.lead_type}")
        logger.info(f"[ENDPOINT] Security filters: agent_id={input_data.agent_id}, agency_id={input_data.agency_id}")
        logger.info(f"[ENDPOINT] Question: {input_data.question[:100]}...")
        
        # Get the restricted agent for this specific lead type
        agent = get_restricted_agent(
            lead_type=input_data.lead_type,
            agent_id=input_data.agent_id,
            agency_id=input_data.agency_id
        )
        
        # Process the question
        logger.info(f"[ENDPOINT] Processing question with agent")
        answer, suggested_questions = agent.process_question(input_data.question)
        
        # Build filter description
        filtered_by = f"agent_id={input_data.agent_id}, agency_id={input_data.agency_id}"
        
        execution_time = round(time.time() - start_time, 2)
        
        logger.info(f"[ENDPOINT] ✓ Query completed successfully in {execution_time}s")
        logger.info(f"[ENDPOINT] Answer length: {len(answer)} characters")
        
        return AncDBResponse(
            question=input_data.question,
            answer=answer,
            sources=[f"{input_data.lead_type} Leads Data", "Customer Data"],
            suggested_questions=suggested_questions,
            model_used="gpt-4o-mini",
            status_code="200",
            found=True,
            execution_time=execution_time,
            filtered_by=filtered_by,
            lead_type=input_data.lead_type
        )
    
    except HTTPException as http_exc:
        logger.error(f"[ENDPOINT] HTTP Exception: {http_exc.detail}")
        raise
    
    except ValueError as ve:
        execution_time = round(time.time() - start_time, 2)
        logger.error(f"[ENDPOINT] Validation error: {str(ve)}")
        
        return AncDBResponse(
            question=input_data.question,
            answer=f"Invalid request: {str(ve)}. Please check your input and try again.",
            sources=[],
            suggested_questions=[
                "Ensure lead_type is one of: Home, Auto, Gas Station, Restaurant, Salon, General Contractor, Shopping Mall, General Business, Hotel/Motel, Tank",
                "Make sure all required fields are provided: question, lead_type, agent_id, agency_id",
                "Check that agent_id and agency_id contain only alphanumeric characters, underscores, and hyphens"
            ],
            model_used="gpt-4o-mini",
            status_code="400",
            found=False,
            execution_time=execution_time,
            filtered_by="Validation failed",
            lead_type=input_data.lead_type if hasattr(input_data, 'lead_type') else ""
        )
    
    except Exception as e:
        execution_time = round(time.time() - start_time, 2)
        logger.error(f"[ENDPOINT] Unexpected error: {type(e).__name__}: {str(e)}")
        logger.debug(f"[ENDPOINT] Stack trace:", exc_info=True)
        
        if "connection" in str(e).lower() or "database" in str(e).lower():
            error_message = "We're experiencing database connectivity issues. Please try again in a moment."
            status_code = "503"
        elif "timeout" in str(e).lower():
            error_message = "Your request took too long to process. Please try a more specific question."
            status_code = "504"
        else:
            error_message = "An unexpected error occurred while processing your request. Please try again."
            status_code = "500"
        
        return AncDBResponse(
            question=input_data.question,
            answer=error_message,
            sources=[],
            suggested_questions=[
                f"Try asking about {input_data.lead_type} leads with customer names",
                "Ask about customer emails or contact information",
                "Inquire about specific customer details by name"
            ],
            model_used="gpt-4o-mini",
            status_code=status_code,
            found=False,
            execution_time=execution_time,
            filtered_by=f"Error: {type(e).__name__}",
            lead_type=input_data.lead_type if hasattr(input_data, 'lead_type') else ""
        )

class DBConnectionResponse(BaseModel):
    status: str
    message: str
    tables_summary: dict = {}
    execution_time: float = 0

@router.get("/db-connection", response_model=DBConnectionResponse)
async def check_db_connection():
    """Check database connection and table access status"""
    start_time = time.time()
    
    try:
        logger.info("[DB CHECK] Starting database connection check")
        
        if not all([DB_HOST, DB_USER, DB_PASSWORD, DB_NAME, DB_PORT]):
            return DBConnectionResponse(
                status="disconnected",
                message="Database configuration incomplete",
                execution_time=round(time.time() - start_time, 3)
            )
        
        encoded_password = quote(DB_PASSWORD)
        connection_url = f"mysql+pymysql://{DB_USER}:{encoded_password}@{DB_HOST}:{DB_PORT}/{DB_NAME}?charset=utf8mb4"
        
        all_tables = list(ALLOWED_TABLES.values()) + ["customer_master"]
        
        db = SQLDatabase.from_uri(connection_url, include_tables=all_tables)
        usable_tables = db.get_usable_table_names()
        
        # Check access for each table
        readable = 0
        writable = 0
        
        for table_name in all_tables:
            if table_name in usable_tables:
                try:
                    db._execute(f"SELECT 1 FROM {table_name} LIMIT 1")
                    readable += 1
                except:
                    pass
        
        # Check write access
        try:
            result = db._execute("SHOW GRANTS")
            for row in result:
                grant_text = str(row[0]).upper()
                if ("INSERT" in grant_text or "UPDATE" in grant_text or "ALL PRIVILEGES" in grant_text):
                    writable = len(all_tables)
                    break
        except:
            pass
        
        execution_time = round(time.time() - start_time, 3)
        
        return DBConnectionResponse(
            status="connected",
            message=f"Connected successfully. {readable}/{len(all_tables)} readable, {writable}/{len(all_tables)} writable",
            tables_summary={
                "total": len(all_tables),
                "readable": readable,
                "writable": writable
            },
            execution_time=execution_time
        )
        
    except Exception as e:
        logger.error(f"[DB CHECK] Connection failed: {str(e)}")
        return DBConnectionResponse(
            status="disconnected",
            message="Database connection failed",
            execution_time=round(time.time() - start_time, 3)
        )