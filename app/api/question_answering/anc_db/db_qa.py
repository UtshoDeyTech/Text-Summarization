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
from .schema_cache_manager import schema_cache
from .query_cache_manager import query_cache
import asyncio

router = APIRouter()

# Startup event flag to ensure initialization happens only once
_schema_cache_initialized = False

# Global database connection pool
_db_connection = None
_connection_url = None

# Global agent pool - stores pre-initialized agents
_agent_pool = {}
_agent_pool_initialized = False

async def initialize_schema_cache_on_startup():
    """
    Initialize schema cache on server startup if not already present.
    This function is called when the application starts.
    """
    global _schema_cache_initialized

    if _schema_cache_initialized:
        logger.info("[STARTUP] Schema cache already initialized, skipping")
        return

    try:
        logger.info("[STARTUP] Checking schema cache status...")

        # Check if cache exists and has data
        cache_info = schema_cache.get_cache_info()

        if cache_info.get("cache_exists") and cache_info.get("total_tables", 0) > 0:
            logger.info(f"[STARTUP] ✓ Schema cache already exists with {cache_info['total_tables']} tables")
            logger.info(f"[STARTUP] Tables cached: {list(cache_info.get('tables', {}).keys())}")
            _schema_cache_initialized = True
            return

        logger.info("[STARTUP] Schema cache is empty or missing, initializing...")

        # Build database connection
        if not all([DB_HOST, DB_USER, DB_PASSWORD, DB_NAME, DB_PORT]):
            logger.warning("[STARTUP] Database configuration incomplete, schema cache initialization skipped")
            return

        encoded_password = quote(DB_PASSWORD)
        connection_url = f"mysql+pymysql://{DB_USER}:{encoded_password}@{DB_HOST}:{DB_PORT}/{DB_NAME}?charset=utf8mb4"

        # Get all tables to cache
        tables_to_cache = list(ALLOWED_TABLES.values()) + ["customer_master"]
        logger.info(f"[STARTUP] Fetching schemas for {len(tables_to_cache)} tables...")

        # Connect to database
        db = SQLDatabase.from_uri(connection_url, include_tables=tables_to_cache)

        schemas_to_save = {}

        # Fetch schema for each table
        for table_name in tables_to_cache:
            try:
                logger.info(f"[STARTUP] Fetching schema for {table_name}")
                table_info = db.get_table_info_no_throw([table_name])

                # Parse column names
                columns = []
                for line in table_info.split('\n'):
                    if line.strip() and not line.strip().startswith('CREATE') and not line.strip().startswith('/*'):
                        match = re.match(r'^\s*`?(\w+)`?\s+', line)
                        if match:
                            columns.append(match.group(1))

                if columns:
                    schemas_to_save[table_name] = {
                        "schema_info": table_info,
                        "column_list": columns
                    }
                    logger.info(f"[STARTUP] ✓ {table_name} - {len(columns)} columns")
                else:
                    logger.warning(f"[STARTUP] No columns found for {table_name}")

            except Exception as table_error:
                logger.error(f"[STARTUP] Error fetching schema for {table_name}: {table_error}")

        # Save all schemas to cache
        if schemas_to_save:
            success = schema_cache.save_all_schemas(schemas_to_save)
            if success:
                logger.info(f"[STARTUP] ✓ Schema cache initialized with {len(schemas_to_save)} tables")
                _schema_cache_initialized = True
            else:
                logger.error(f"[STARTUP] Failed to save schemas to cache")
        else:
            logger.warning(f"[STARTUP] No schemas were fetched successfully")

    except Exception as e:
        logger.error(f"[STARTUP] Error initializing schema cache: {str(e)}")
        # Don't raise exception - allow server to start even if cache init fails

def initialize_database_connection():
    """
    Initialize database connection at startup.
    This connection is reused across all requests.
    """
    global _db_connection, _connection_url

    try:
        logger.info("[STARTUP] Initializing database connection pool...")

        if not all([DB_HOST, DB_USER, DB_PASSWORD, DB_NAME, DB_PORT]):
            logger.warning("[STARTUP] Database configuration incomplete, connection pool not initialized")
            return False

        # Build connection URL (pool parameters handled by SQLAlchemy engine, not URL)
        encoded_password = quote(DB_PASSWORD)
        _connection_url = f"mysql+pymysql://{DB_USER}:{encoded_password}@{DB_HOST}:{DB_PORT}/{DB_NAME}?charset=utf8mb4"

        # Get all tables
        all_tables = list(ALLOWED_TABLES.values()) + ["customer_master"]

        # Create database connection
        _db_connection = SQLDatabase.from_uri(
            _connection_url,
            include_tables=all_tables,
            sample_rows_in_table_info=2
        )

        logger.info(f"[STARTUP] ✓ Database connection pool initialized")
        logger.info(f"[STARTUP] ✓ Connected to {len(all_tables)} tables")
        return True

    except Exception as e:
        logger.error(f"[STARTUP] Failed to initialize database connection: {str(e)}")
        return False


def initialize_agent_pool():
    """
    Initialize agent pool at startup.
    Creates one agent instance for each lead type that can be reused.

    Note: Agents are created without specific agent_id/agency_id filtering.
    The filtering is applied at query time through the system prompt.
    """
    global _agent_pool, _agent_pool_initialized

    if _agent_pool_initialized:
        logger.info("[STARTUP] Agent pool already initialized, skipping")
        return True

    try:
        logger.info("[STARTUP] Initializing agent pool...")

        if not _db_connection:
            logger.warning("[STARTUP] Database connection not available, skipping agent pool initialization")
            return False

        # Create LLM instance (shared across all agents) - optimized for speed
        llm = ChatOpenAI(
            model_name="gpt-4o-mini",
            temperature=0,
            max_tokens=800,  # Reduced for faster responses
            request_timeout=SecurityConfig.MAX_EXECUTION_TIME
        )

        # For each lead type, create a base agent configuration
        # We'll store the toolkit and LLM, and create agents on-demand with specific filters
        for lead_type, table_name in ALLOWED_TABLES.items():
            try:
                logger.info(f"[STARTUP] Creating agent toolkit for {lead_type} ({table_name})")

                # Create a database connection for this specific table
                table_db = SQLDatabase.from_uri(
                    _connection_url,
                    include_tables=[table_name, "customer_master"],
                    sample_rows_in_table_info=2
                )

                # Store the database connection and LLM for this lead type
                # We'll create the actual agent when needed with specific agent_id/agency_id
                _agent_pool[lead_type] = {
                    "db": table_db,
                    "llm": llm,
                    "table_name": table_name
                }

                logger.info(f"[STARTUP] ✓ Toolkit ready for {lead_type}")

            except Exception as table_error:
                logger.error(f"[STARTUP] Failed to initialize toolkit for {lead_type}: {table_error}")

        _agent_pool_initialized = True
        logger.info(f"[STARTUP] ✓ Agent pool initialized with {len(_agent_pool)} lead types")
        return True

    except Exception as e:
        logger.error(f"[STARTUP] Failed to initialize agent pool: {str(e)}")
        return False


@router.on_event("startup")
async def startup_event():
    """FastAPI startup event handler"""
    logger.info("[STARTUP] ========================================")
    logger.info("[STARTUP] Starting ANC DB service initialization")
    logger.info("[STARTUP] ========================================")

    # Step 1: Initialize schema cache
    logger.info("[STARTUP] Step 1: Schema cache initialization...")
    await initialize_schema_cache_on_startup()
    logger.info("[STARTUP] ✓ Schema cache initialization complete")

    # Step 2: Initialize database connection pool
    logger.info("[STARTUP] Step 2: Database connection pool initialization...")
    db_success = initialize_database_connection()
    if db_success:
        logger.info("[STARTUP] ✓ Database connection pool initialized")
    else:
        logger.warning("[STARTUP] ✗ Database connection pool initialization failed")

    # Step 3: Initialize agent pool
    logger.info("[STARTUP] Step 3: Agent pool initialization...")
    agent_success = initialize_agent_pool()
    if agent_success:
        logger.info("[STARTUP] ✓ Agent pool initialized")
    else:
        logger.warning("[STARTUP] ✗ Agent pool initialization failed")

    logger.info("[STARTUP] ========================================")
    logger.info("[STARTUP] ANC DB service initialization complete")
    logger.info("[STARTUP] ========================================")

# Security Configuration
class SecurityConfig:
    MAX_QUERY_LENGTH = 3000
    MAX_ITERATIONS = 5  # Max 5, but agent should complete in 1-2 iterations with optimized prompts
    MAX_EXECUTION_TIME = 45  # Reduced from 60 to 45 - faster timeout
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
    Convert text responses containing multiple items into HTML list format (optimized).
    Detects numbered lists, bullet points, and multiple distinct items.
    """
    if not text or len(text.strip()) < 10:
        return text

    text = text.strip()

    # Pattern 1: Detect numbered lists (1. item, 2. item, etc.)
    numbered_pattern = re.compile(r'^(\d+\.\s+.+?)(?=\n\d+\.|$)', re.MULTILINE | re.DOTALL)
    numbered_matches = numbered_pattern.findall(text)

    if len(numbered_matches) >= 2:
        # Extract items without backslash in f-string
        items = []
        for m in numbered_matches:
            if m.strip():
                clean_item = re.sub(r'^\d+\.\s*', '', m.strip())
                items.append(f"<li>{clean_item}</li>")
        if items:
            return f"<ol>{''.join(items)}</ol>"

    # Pattern 2: Detect bullet points (-, *, •, etc.)
    bullet_pattern = re.compile(r'^([-*•]\s+.+?)(?=\n[-*•]|$)', re.MULTILINE | re.DOTALL)
    bullet_matches = bullet_pattern.findall(text)

    if len(bullet_matches) >= 2:
        # Extract items without backslash in f-string
        items = []
        for m in bullet_matches:
            if m.strip():
                clean_item = re.sub(r'^[-*•]\s*', '', m.strip())
                items.append(f"<li>{clean_item}</li>")
        if items:
            return f"<ul>{''.join(items)}</ul>"

    # Pattern 3: Detect multiple distinct lines (emails, phones, names)
    lines = [line.strip() for line in text.split('\n') if line.strip()]

    if 2 <= len(lines) <= 10:
        email_count = sum(1 for line in lines if '@' in line)
        if email_count >= len(lines) * 0.6 or all(10 < len(line) < 200 for line in lines):
            items = [f"<li>{line}</li>" for line in lines]
            return f"<ul>{''.join(items)}</ul>"

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

CRITICAL INSTRUCTIONS - SINGLE ITERATION REQUIRED:
1. Schema is PROVIDED ABOVE - DO NOT call sql_db_schema or sql_db_list_tables
2. You already know the tables: {table_name} and customer_master
3. Review the schema above and construct your SQL query IMMEDIATELY
4. Execute query with sql_db_query tool in your FIRST action
5. IMPORTANT: Do NOT explore - EXECUTE IMMEDIATELY in iteration 1

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

SINGLE-ITERATION WORKFLOW (MAXIMUM SPEED):
ITERATION 1 ONLY - Execute sql_db_query immediately:
1. Read question + schema above (already loaded - no tools needed)
2. Construct SQL with security filters
3. Call sql_db_query tool RIGHT NOW
4. Format result and respond

DO NOT:
- Call sql_db_list_tables (you know: {table_name}, customer_master)
- Call sql_db_schema (schema is provided above)
- Call sql_db_query_checker (skip validation for speed)
- Take multiple iterations (EXECUTE IN ITERATION 1!)

SECURITY REMINDERS:
- NEVER query tables other than {table_name} and customer_master
- ALWAYS apply security filters on {table_name}
- agent_id = '{agent_id}' AND agency_id = '{agency_id}' must be in WHERE clause for {table_name}
- Only use SELECT queries - no modifications allowed"""

class RestrictedAncDBAgent:
    def __init__(self, lead_type: str, agent_id: str, agency_id: str, use_pool: bool = True):
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
        self.use_pool = use_pool

        logger.info(f"[AGENT INIT] Tables allowed: {self.table_name}, customer_master")
        logger.info(f"[AGENT INIT] Security filters: agent_id={self.agent_id}, agency_id={self.agency_id}")
        logger.info(f"[AGENT INIT] Using pool: {self.use_pool}")

        # Check if we can use the pool
        if self.use_pool and lead_type in _agent_pool:
            logger.info(f"[AGENT INIT] ✓ Using pooled resources for {lead_type}")
            self.db = _agent_pool[lead_type]["db"]
            self.llm = _agent_pool[lead_type]["llm"]
            self.connection_url = _connection_url
        else:
            # Fallback to creating new connection
            logger.info(f"[AGENT INIT] Pool not available, creating new connection")
            self.setup_environment()
            self.connection_url = self.build_connection_url()
            self.db = None
            self.llm = None

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

            # If using pool, db is already set
            if not self.db:
                # Include both the lead table and customer_master
                self.db = SQLDatabase.from_uri(
                    self.connection_url,
                    include_tables=[self.table_name, "customer_master"],
                    sample_rows_in_table_info=2  # Show sample data in schema
                )
                connection_time = round(time.time() - start_time, 3)
                logger.info(f"[DB CONNECTION] ✓ Database connected successfully in {connection_time}s")
            else:
                connection_time = round(time.time() - start_time, 3)
                logger.info(f"[DB CONNECTION] ✓ Using pooled database connection ({connection_time}s)")

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
        """Get a formatted summary of the table schema from cache or database"""
        try:
            # Try to get schema from cache first
            logger.debug(f"[SCHEMA] Checking cache for {self.table_name}")
            cached_schema = schema_cache.get_formatted_schema(self.table_name)

            if cached_schema:
                logger.info(f"[SCHEMA] ✓ Using cached schema for {self.table_name}")
                return cached_schema

            # If not in cache, fetch from database
            logger.info(f"[SCHEMA] Cache miss, fetching schema from database for {self.table_name}")
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

            # Save to cache for future use
            if columns:
                logger.info(f"[SCHEMA] Saving schema to cache for {self.table_name} ({len(columns)} columns)")
                schema_cache.save_schema(self.table_name, table_info, columns)

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

            # If using pool, LLM is already set
            if not self.llm:
                logger.debug("[AGENT INIT] Creating ChatOpenAI LLM instance")
                self.llm = ChatOpenAI(
                    model_name="gpt-4o-mini",
                    temperature=0,
                    max_tokens=800,  # Reduced from 2000 to 800 - faster responses, most answers are short
                    request_timeout=SecurityConfig.MAX_EXECUTION_TIME
                )
                logger.info(f"[AGENT INIT] ✓ LLM initialized (model: gpt-4o-mini)")
            else:
                logger.info(f"[AGENT INIT] ✓ Using pooled LLM (model: gpt-4o-mini)")
            
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
                verbose=False,  # Disabled verbose mode for faster execution
                agent_type="openai-tools",
                system_message=system_message,
                max_iterations=SecurityConfig.MAX_ITERATIONS,
                handle_parsing_errors=True
                # Removed early_stopping_method - not supported in this LangChain version
            )
            
            init_time = round(time.time() - start_time, 3)
            logger.info(f"[AGENT INIT] ✓ Agent initialized successfully in {init_time}s")
            return True
            
        except Exception as e:
            logger.error(f"[AGENT INIT] ✗ Agent initialization failed: {type(e).__name__}: {str(e)}")
            return False
    
    def generate_no_results_message(self, question: str) -> str:
        """Generate a natural, AI-created message when no results are found (optimized)"""
        try:
            logger.debug("[NO RESULTS] Generating AI-powered 'no results' message")

            no_results_llm = ChatOpenAI(
                model_name="gpt-4o-mini",
                temperature=0.8,
                max_tokens=80  # Reduced from 150 to 80 - concise messages
            )

            # Shorter prompt for faster response
            prompt = f"""User asked: "{question}" about {self.lead_type} leads. No data found.
Write a friendly 2-sentence response (no table names):"""

            response = no_results_llm.invoke(prompt)
            message = response.content if hasattr(response, 'content') else str(response)

            logger.info(f"[NO RESULTS] Generated message: {message[:100]}...")
            return message.strip()

        except Exception as e:
            logger.error(f"[NO RESULTS] Error generating message: {e}")
            return f"I couldn't find any relevant information in your {self.lead_type} leads for that query. Please try rephrasing your question or providing more specific details."
    
    def generate_natural_suggestions(self, question: str, found_data: bool) -> List[str]:
        """Generate natural, contextual suggested questions using AI (optimized)"""
        try:
            logger.debug("[SUGGESTIONS] Generating AI-powered suggestions")

            suggestions_llm = ChatOpenAI(
                model_name="gpt-4o-mini",
                temperature=0.7,
                max_tokens=100  # Reduced from 200 to 100 - suggestions are short
            )

            data_context = "data was found" if found_data else "no data was found"

            # Shorter, more direct prompt
            prompt = f"""User asked: "{question}" about {self.lead_type} leads ({data_context}).
Generate 3 follow-up questions (under 12 words each, one per line, no numbers):"""

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
    
    def generate_sql_directly(self, question: str) -> str:
        """
        Generate SQL query directly using LLM without agent iterations.
        This is MUCH faster than letting the agent explore.
        """
        try:
            logger.info(f"[DIRECT SQL] Generating SQL query for: {question[:100]}...")

            # Get schema
            schema_summary = self.get_table_schema_summary()
            filter_clause = f"{self.table_name}.agent_id = '{self.agent_id}' AND {self.table_name}.agency_id = '{self.agency_id}'"

            # Create a focused prompt for SQL generation only
            sql_generation_prompt = f"""Generate a SQL query for this question.

QUESTION: {question}

TABLES:
- {self.table_name} AS l (alias 'l') - Lead data
- customer_master AS cm (alias 'cm') - Customer info

SCHEMA FOR {self.table_name}:
{schema_summary}

MANDATORY FILTER (use table alias 'l'):
WHERE l.agent_id = '{self.agent_id}' AND l.agency_id = '{self.agency_id}'

TEMPLATE (copy this and modify):
```sql
SELECT cm.first_name, cm.last_name, cm.email_id, l.email, l.mobile_number
FROM {self.table_name} l
INNER JOIN customer_master cm ON l.customer_id = cm.customer_id
WHERE l.agent_id = '{self.agent_id}' AND l.agency_id = '{self.agency_id}'
AND (cm.first_name LIKE '%<name>%' OR cm.last_name LIKE '%<name>%')
LIMIT 10
```

CRITICAL RULES:
1. Use table aliases: 'l' for {self.table_name}, 'cm' for customer_master
2. Security filter MUST use 'l' alias: WHERE l.agent_id = '{self.agent_id}' AND l.agency_id = '{self.agency_id}'
3. For customer names → use INNER JOIN with customer_master
4. Extract search term from question and use in LIKE clause
5. ONLY output the SQL query, nothing else

SQL QUERY:"""

            # Use a fast LLM call to generate SQL
            sql_generator = ChatOpenAI(
                model_name="gpt-4o-mini",
                temperature=0,
                max_tokens=300
            )

            response = sql_generator.invoke(sql_generation_prompt)
            sql_query = response.content if hasattr(response, 'content') else str(response)

            # Extract SQL from markdown code blocks if present
            if "```sql" in sql_query:
                sql_query = sql_query.split("```sql")[1].split("```")[0].strip()
            elif "```" in sql_query:
                sql_query = sql_query.split("```")[1].split("```")[0].strip()

            logger.info(f"[DIRECT SQL] Generated query: {sql_query[:200]}...")
            return sql_query

        except Exception as e:
            logger.error(f"[DIRECT SQL] Error generating SQL: {e}")
            return None

    def execute_sql_directly(self, sql_query: str) -> str:
        """Execute SQL query directly against database"""
        try:
            logger.info(f"[DIRECT EXEC] Executing query...")
            result = self.db.run(sql_query)
            logger.info(f"[DIRECT EXEC] Query executed successfully, result length: {len(str(result))}")
            return str(result)
        except Exception as e:
            logger.error(f"[DIRECT EXEC] Query execution error: {e}")
            return f"Error executing query: {str(e)}"

    def format_sql_results(self, question: str, sql_result: str) -> str:
        """Format SQL results into natural language answer"""
        try:
            logger.info(f"[FORMAT] Formatting SQL results into natural language...")

            formatter_prompt = f"""Convert this SQL query result into a natural, conversational answer.

USER QUESTION: {question}
SQL RESULT: {sql_result}

INSTRUCTIONS:
1. Write a clear, concise answer (1-2 sentences max)
2. Use natural language, not technical terms
3. If multiple results, list them clearly
4. If no results, say "No information found"
5. NEVER mention table names or technical details

ANSWER:"""

            formatter = ChatOpenAI(
                model_name="gpt-4o-mini",
                temperature=0,
                max_tokens=200
            )

            response = formatter.invoke(formatter_prompt)
            answer = response.content if hasattr(response, 'content') else str(response)

            logger.info(f"[FORMAT] Formatted answer: {answer[:100]}...")
            return answer.strip()

        except Exception as e:
            logger.error(f"[FORMAT] Error formatting results: {e}")
            return sql_result

    def process_question_direct(self, question: str) -> tuple:
        """
        Process question using DIRECT SQL generation (fast path).
        Bypasses agent exploration entirely.
        """
        logger.info(f"[DIRECT MODE] Processing question: {question[:100]}...")
        logger.info(f"[DIRECT MODE] Lead type: {self.lead_type}, Table: {self.table_name}")

        try:
            start_time = time.time()

            # Step 1: Generate SQL directly (1 LLM call)
            sql_query = self.generate_sql_directly(question)
            if not sql_query:
                return ("Error generating SQL query", [])

            sql_gen_time = round(time.time() - start_time, 3)
            logger.info(f"[DIRECT MODE] SQL generation took {sql_gen_time}s")

            # Step 2: Execute SQL
            sql_result = self.execute_sql_directly(sql_query)
            exec_time = round(time.time() - start_time, 3)
            logger.info(f"[DIRECT MODE] SQL execution took {exec_time - sql_gen_time}s")

            # Step 3: Format results (1 LLM call)
            answer = self.format_sql_results(question, sql_result)
            format_time = round(time.time() - start_time, 3)
            logger.info(f"[DIRECT MODE] Formatting took {format_time - exec_time}s")

            # Step 4: Generate suggestions (1 LLM call)
            suggestions = self.generate_natural_suggestions(question, found_data=True)

            total_time = round(time.time() - start_time, 3)
            logger.info(f"[DIRECT MODE] ✓ Total time: {total_time}s (SQL: {sql_gen_time}s, Exec: {exec_time - sql_gen_time}s, Format: {format_time - exec_time}s)")

            formatted_answer = format_response_as_html_list(answer)
            return (formatted_answer, suggestions[:3])

        except Exception as e:
            logger.error(f"[DIRECT MODE] Error: {type(e).__name__}: {str(e)}")
            raise

    def process_question(self, question: str) -> tuple:
        logger.info(f"[PROCESS] Processing question: {question[:100]}...")
        logger.info(f"[PROCESS] Lead type: {self.lead_type}, Table: {self.table_name}")
        logger.info(f"[PROCESS] Security filters: agent_id={self.agent_id}, agency_id={self.agency_id}")

        try:
            filter_clause = f"{self.table_name}.agent_id = '{self.agent_id}' AND {self.table_name}.agency_id = '{self.agency_id}'"

            # Get schema summary upfront to avoid schema tool calls
            schema_summary = self.get_table_schema_summary()

            # Create example SQL based on question type
            example_sql = f"""SELECT cm.first_name, cm.last_name, cm.email_id, l.email, l.mobile_number
FROM {self.table_name} l
INNER JOIN customer_master cm ON l.customer_id = cm.customer_id
WHERE {filter_clause}
AND (cm.first_name LIKE '%<name>%' OR cm.last_name LIKE '%<name>%')
LIMIT 10"""

            prompt = f"""Question: {question}

TABLES AVAILABLE:
- {self.table_name} (70+ columns - schema below)
- customer_master (customer_id, first_name, last_name, email_id, mobile_number, work_number)

SCHEMA FOR {self.table_name}:
{schema_summary}

⚡ YOUR FIRST ACTION MUST BE: sql_db_query

EXAMPLE QUERY TEMPLATE:
{example_sql}

Replace <name> with the actual search term from the question.
Modify SELECT clause based on what user asks for.

MANDATORY SECURITY FILTER:
WHERE {filter_clause}

EXECUTE IMMEDIATELY - Do NOT call:
- sql_db_list_tables (tables already listed above)
- sql_db_schema (schema already provided above)
- sql_db_query_checker (skip validation)

ACTION REQUIRED NOW:
1. Review question: "{question}"
2. Use schema above to find matching columns
3. Build SQL query with security filters
4. Execute sql_db_query immediately
5. Return formatted answer

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
                "not available", "doesn't exist", "no such column", "no data",
                "agent stopped", "max iterations"
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

        # Check query cache first to avoid API calls
        cached_result = query_cache.get_cached_result(
            question=input_data.question,
            lead_type=input_data.lead_type,
            agent_id=input_data.agent_id,
            agency_id=input_data.agency_id
        )

        if cached_result:
            answer, suggested_questions = cached_result
            execution_time = round(time.time() - start_time, 2)

            logger.info(f"[ENDPOINT] ✓ Query served from cache in {execution_time}s (no API calls!)")

            return AncDBResponse(
                question=input_data.question,
                answer=answer,
                sources=[f"{input_data.lead_type} Leads Data (cached)", "Customer Data"],
                suggested_questions=suggested_questions,
                model_used="gpt-4o-mini (cached)",
                status_code="200",
                found=True,
                execution_time=execution_time,
                filtered_by=f"agent_id={input_data.agent_id}, agency_id={input_data.agency_id}",
                lead_type=input_data.lead_type
            )

        # Cache miss - process with agent
        logger.info(f"[ENDPOINT] Cache miss, processing with agent...")

        # Get the restricted agent for this specific lead type
        agent = get_restricted_agent(
            lead_type=input_data.lead_type,
            agent_id=input_data.agent_id,
            agency_id=input_data.agency_id
        )

        # Process the question using DIRECT MODE (bypasses agent iterations)
        logger.info(f"[ENDPOINT] Processing question with DIRECT SQL generation")
        answer, suggested_questions = agent.process_question_direct(input_data.question)

        # Cache the result for future requests (async - don't wait)
        asyncio.create_task(asyncio.to_thread(
            query_cache.cache_result,
            question=input_data.question,
            lead_type=input_data.lead_type,
            agent_id=input_data.agent_id,
            agency_id=input_data.agency_id,
            answer=answer,
            suggested_questions=suggested_questions
        ))
        
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

class SchemaRefreshRequest(BaseModel):
    table_name: Optional[str] = Field(None, description="Specific table to refresh, or leave empty to refresh all tables")

class SchemaRefreshResponse(BaseModel):
    status: str
    message: str
    tables_refreshed: List[str] = []
    execution_time: float = 0
    cache_info: dict = {}

@router.post("/refresh-schema-cache", response_model=SchemaRefreshResponse)
async def refresh_schema_cache(request: SchemaRefreshRequest = None):
    """
    Refresh the schema cache by fetching the latest table structures from the database.

    This endpoint should be called when:
    - Tables are modified (columns added, removed, or renamed)
    - Initial setup of the application
    - Periodic maintenance

    Args:
        table_name: Optional specific table to refresh. If not provided, all tables will be refreshed.

    Returns:
        Status of the refresh operation including tables refreshed and cache information
    """
    start_time = time.time()

    try:
        logger.info("[SCHEMA REFRESH] Starting schema cache refresh")

        # Determine which tables to refresh
        if request and request.table_name:
            # Validate table name
            table_name = request.table_name
            if table_name not in ALLOWED_TABLES.values() and table_name != "customer_master":
                return SchemaRefreshResponse(
                    status="error",
                    message=f"Invalid table name: {table_name}",
                    execution_time=round(time.time() - start_time, 3)
                )
            tables_to_refresh = [table_name]
            logger.info(f"[SCHEMA REFRESH] Refreshing specific table: {table_name}")
        else:
            # Refresh all tables
            tables_to_refresh = list(ALLOWED_TABLES.values()) + ["customer_master"]
            logger.info(f"[SCHEMA REFRESH] Refreshing all {len(tables_to_refresh)} tables")

        # Build database connection
        if not all([DB_HOST, DB_USER, DB_PASSWORD, DB_NAME, DB_PORT]):
            return SchemaRefreshResponse(
                status="error",
                message="Database configuration incomplete",
                execution_time=round(time.time() - start_time, 3)
            )

        encoded_password = quote(DB_PASSWORD)
        connection_url = f"mysql+pymysql://{DB_USER}:{encoded_password}@{DB_HOST}:{DB_PORT}/{DB_NAME}?charset=utf8mb4"

        # Connect to database
        db = SQLDatabase.from_uri(connection_url, include_tables=tables_to_refresh)

        refreshed_tables = []
        schemas_to_save = {}

        # Fetch schema for each table
        for table_name in tables_to_refresh:
            try:
                logger.info(f"[SCHEMA REFRESH] Fetching schema for {table_name}")
                table_info = db.get_table_info_no_throw([table_name])

                # Parse column names
                columns = []
                for line in table_info.split('\n'):
                    if line.strip() and not line.strip().startswith('CREATE') and not line.strip().startswith('/*'):
                        match = re.match(r'^\s*`?(\w+)`?\s+', line)
                        if match:
                            columns.append(match.group(1))

                if columns:
                    schemas_to_save[table_name] = {
                        "schema_info": table_info,
                        "column_list": columns
                    }
                    refreshed_tables.append(table_name)
                    logger.info(f"[SCHEMA REFRESH] ✓ {table_name} - {len(columns)} columns")
                else:
                    logger.warning(f"[SCHEMA REFRESH] No columns found for {table_name}")

            except Exception as table_error:
                logger.error(f"[SCHEMA REFRESH] Error fetching schema for {table_name}: {table_error}")

        # Save all schemas to cache
        if schemas_to_save:
            success = schema_cache.save_all_schemas(schemas_to_save)
            if success:
                logger.info(f"[SCHEMA REFRESH] ✓ Successfully refreshed {len(refreshed_tables)} tables")
            else:
                logger.error(f"[SCHEMA REFRESH] Failed to save schemas to cache")
                return SchemaRefreshResponse(
                    status="error",
                    message="Failed to save schemas to cache",
                    tables_refreshed=refreshed_tables,
                    execution_time=round(time.time() - start_time, 3)
                )

        # Get cache info
        cache_info = schema_cache.get_cache_info()

        execution_time = round(time.time() - start_time, 3)

        return SchemaRefreshResponse(
            status="success",
            message=f"Successfully refreshed {len(refreshed_tables)} table(s)",
            tables_refreshed=refreshed_tables,
            execution_time=execution_time,
            cache_info=cache_info
        )

    except Exception as e:
        logger.error(f"[SCHEMA REFRESH] Error: {str(e)}")
        return SchemaRefreshResponse(
            status="error",
            message=f"Schema refresh failed: {str(e)}",
            execution_time=round(time.time() - start_time, 3)
        )

class SchemaCacheInfoResponse(BaseModel):
    status: str
    cache_info: dict
    execution_time: float = 0

@router.get("/schema-cache-info", response_model=SchemaCacheInfoResponse)
async def get_schema_cache_info():
    """
    Get information about the current schema cache

    Returns:
        Cache information including tables cached and last updated times
    """
    start_time = time.time()

    try:
        logger.info("[SCHEMA CACHE INFO] Retrieving cache information")
        cache_info = schema_cache.get_cache_info()

        execution_time = round(time.time() - start_time, 3)

        return SchemaCacheInfoResponse(
            status="success",
            cache_info=cache_info,
            execution_time=execution_time
        )

    except Exception as e:
        logger.error(f"[SCHEMA CACHE INFO] Error: {str(e)}")
        return SchemaCacheInfoResponse(
            status="error",
            cache_info={},
            execution_time=round(time.time() - start_time, 3)
        )

class QueryCacheInfoResponse(BaseModel):
    status: str
    cache_info: dict
    execution_time: float = 0

@router.get("/query-cache-info", response_model=QueryCacheInfoResponse)
async def get_query_cache_info():
    """
    Get information about the query result cache

    Returns:
        Cache information including entry count, active entries, and statistics
    """
    start_time = time.time()

    try:
        logger.info("[QUERY CACHE INFO] Retrieving cache information")
        cache_info = query_cache.get_cache_info()

        execution_time = round(time.time() - start_time, 3)

        return QueryCacheInfoResponse(
            status="success",
            cache_info=cache_info,
            execution_time=execution_time
        )

    except Exception as e:
        logger.error(f"[QUERY CACHE INFO] Error: {str(e)}")
        return QueryCacheInfoResponse(
            status="error",
            cache_info={},
            execution_time=round(time.time() - start_time, 3)
        )

class QueryCacheClearRequest(BaseModel):
    lead_type: Optional[str] = Field(None, description="Specific lead type to clear")
    agent_id: Optional[str] = Field(None, description="Specific agent ID to clear")

class QueryCacheClearResponse(BaseModel):
    status: str
    message: str
    execution_time: float = 0

@router.post("/clear-query-cache", response_model=QueryCacheClearResponse)
async def clear_query_cache(request: QueryCacheClearRequest = None):
    """
    Clear the query result cache

    Args:
        lead_type: Optional - clear only entries for specific lead type
        agent_id: Optional - clear only entries for specific agent

    Returns:
        Status of the clear operation
    """
    start_time = time.time()

    try:
        if request and (request.lead_type or request.agent_id):
            logger.info(f"[QUERY CACHE CLEAR] Clearing cache for lead_type={request.lead_type}, agent_id={request.agent_id}")
            query_cache.clear_cache(lead_type=request.lead_type, agent_id=request.agent_id)
            message = f"Cleared cache for lead_type={request.lead_type}, agent_id={request.agent_id}"
        else:
            logger.info(f"[QUERY CACHE CLEAR] Clearing all cache")
            query_cache.clear_cache()
            message = "All query cache cleared"

        execution_time = round(time.time() - start_time, 3)

        return QueryCacheClearResponse(
            status="success",
            message=message,
            execution_time=execution_time
        )

    except Exception as e:
        logger.error(f"[QUERY CACHE CLEAR] Error: {str(e)}")
        return QueryCacheClearResponse(
            status="error",
            message=f"Failed to clear cache: {str(e)}",
            execution_time=round(time.time() - start_time, 3)
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