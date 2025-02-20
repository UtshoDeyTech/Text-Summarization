from fastapi import APIRouter, HTTPException
from typing import List
from pydantic import BaseModel, Field, validator
from config import S3_BUCKET_NAME
from app.service.log_client import logger
from app.service import s3_storage
from langchain_experimental.agents.agent_toolkits import create_pandas_dataframe_agent
from langchain_openai import ChatOpenAI
import pandas as pd
import time
from enum import Enum
import io
from urllib.parse import unquote

router = APIRouter()

class FileType(str, Enum):
    CSV = "csv"
    XLSX = "xlsx"
    XLS = "xls"

class S3URLInput(BaseModel):
    url: str = Field(description="S3 URL of the CSV or Excel file to process")
    question: str = Field(description="Question to be answered about the data")
    
    @validator('url')
    def validate_s3_url(cls, v):
        if not v.startswith(f"https://{S3_BUCKET_NAME}.s3."):
            raise ValueError("Invalid S3 URL format")
        
        lower_url = v.lower()
        if not any(lower_url.endswith(f".{ext}") for ext in [e.value for e in FileType]):
            raise ValueError("URL must point to a CSV or Excel file")
        return v

    def get_file_type(self) -> FileType:
        lower_url = self.url.lower()
        for file_type in FileType:
            if lower_url.endswith(f".{file_type.value}"):
                return file_type
        raise ValueError("Unsupported file type")

class TableQAResponse(BaseModel):
    question: str
    answer: str
    sources: List[str] = []
    suggested_questions: List[str] = []
    model_used: str
    status_code: str = "200"
    found: bool = True
    execution_time: float = 0

# System and user prompts remain the same
SYSTEM_PROMPT = """You are an expert data analyst assistant. When answering questions:
- Provide direct, concise answers that focus exactly on what was asked
- Use natural, conversational language
- Include relevant numbers and statistics when appropriate
- Format numbers clearly with commas for readability
- Make sure your answer directly addresses the specific question

After providing the answer, suggest 3 relevant follow-up questions that would be useful for deeper analysis. These questions should:
- Be completely different from the current question
- Focus on exploring new aspects of the data
- Help understand different patterns or relationships
- Lead to additional insights about the data
- Never repeat or rephrase the current question"""

USER_PROMPT_TEMPLATE = """Based on the data provided, please answer this question: {question}

After your answer, list exactly 3 suggested follow-up questions that would be valuable to ask next. 
Important: The suggested questions should be different from the current question and should help explore the data further.

Format your response as:
ANSWER: [Your clear, concise answer here]
SUGGESTED_QUESTIONS:
1. [First follow-up question - must be different from the current question]
2. [Second follow-up question - must be different from the current question]
3. [Third follow-up question - must be different from the current question]"""

async def get_file_from_s3(url: str) -> bytes:
    try:
        base_url = f"https://{S3_BUCKET_NAME}.s3.us-east-1.amazonaws.com/"
        if base_url not in url:
            raise ValueError("Invalid S3 URL format")
            
        object_key = unquote(url.replace(base_url, ""))
        logger.info(f"Extracted object key: {object_key}")
        
        if not s3_storage.verify_bucket(S3_BUCKET_NAME):
            raise HTTPException(
                status_code=404,
                detail="S3 bucket not accessible"
            )
        
        response = s3_storage.s3_client.get_object(Bucket=S3_BUCKET_NAME, Key=object_key)
        return response['Body'].read()
        
    except Exception as e:
        logger.error(f"Error fetching from S3: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to fetch from S3: {str(e)}"
        )

@router.post("/table_qa", response_model=TableQAResponse)
async def table_qa(input_data: S3URLInput):
    start_time = time.time()
    
    try:
        # Get file type and content from S3
        file_type = input_data.get_file_type()
        file_content = await get_file_from_s3(input_data.url)
        
        try:
            # Read the file into a pandas DataFrame with error handling
            try:
                if file_type == FileType.CSV:
                    df = pd.read_csv(io.BytesIO(file_content))
                else:  # XLSX or XLS
                    df = pd.read_excel(io.BytesIO(file_content))
                
                if df.empty:
                    raise ValueError("The file appears to be empty")
                    
            except Exception as e:
                logger.error(f"Error reading file: {str(e)}")
                raise HTTPException(
                    status_code=400,
                    detail=f"Unable to read file content: {str(e)}"
                )

            # Initialize ChatOpenAI with error handling
            try:
                llm = ChatOpenAI(
                    model="gpt-3.5-turbo",
                    temperature=0
                )
            except Exception as e:
                logger.error(f"Error initializing LLM: {str(e)}")
                raise HTTPException(
                    status_code=500,
                    detail=f"Failed to initialize AI model: {str(e)}"
                )

            # Create pandas DataFrame agent
            agent = create_pandas_dataframe_agent(
                llm=llm,
                df=df,
                agent_type="tool-calling",
                verbose=True,
                allow_dangerous_code=True,
                include_df_in_prompt=True,
                number_of_head_rows=5,
                max_iterations=15,
                early_stopping_method="force",
                prefix=SYSTEM_PROMPT
            )
            
            # Store original question and format it
            original_question = input_data.question.strip()
            formatted_question = USER_PROMPT_TEMPLATE.format(question=original_question)
            
            # Get response from agent
            response = agent.invoke({"input": formatted_question})
            raw_output = response.get("output", "")

            # Parse the response
            answer_parts = raw_output.split("SUGGESTED_QUESTIONS:")
            main_answer = answer_parts[0].replace("ANSWER:", "").strip()
            suggested_questions = []
            
            if len(answer_parts) > 1:
                questions_text = answer_parts[1].strip()
                for line in questions_text.split("\n"):
                    if line.strip() and any(line.strip().startswith(str(i)) for i in range(1, 4)):
                        question = line.strip().split(".", 1)[1].strip()
                        suggested_questions.append(question)

            return TableQAResponse(
                question=original_question,
                answer=main_answer,
                sources=[input_data.url],
                suggested_questions=suggested_questions,
                model_used="GPT-3.5-turbo",
                status_code="200",
                found=True,
                execution_time=round(time.time() - start_time, 2)
            )

        except Exception as e:
            logger.error(f"Error processing file: {str(e)}")
            raise HTTPException(
                status_code=400,
                detail=f"Error processing file: {str(e)}"
            )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error in table_qa: {str(e)}")
        execution_time = time.time() - start_time
        return TableQAResponse(
            question=input_data.question,
            answer="",
            sources=[],
            suggested_questions=[],
            model_used="GPT-3.5-turbo",
            status_code="500",
            found=False,
            execution_time=round(execution_time, 2)
        )