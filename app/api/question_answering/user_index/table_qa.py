from fastapi import APIRouter, HTTPException, File, UploadFile, Form
from typing import List
from pydantic import BaseModel
import tempfile
import os
from app.service.log_client import logger
from langchain_experimental.agents.agent_toolkits import create_pandas_dataframe_agent
from langchain_openai import ChatOpenAI
import pandas as pd
import time

router = APIRouter()

class TableQAResponse(BaseModel):
    question: str
    answer: str
    sources: List[str] = []
    suggested_questions: List[str] = []
    model_used: str
    status_code: str = "200"
    found: bool = True
    execution_time: float = 0

# Define system and user prompts
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

@router.post("/table_qa", response_model=TableQAResponse)
async def table_qa(
    file: UploadFile = File(...),
    question: str = Form(...)
):
    start_time = time.time()
    
    try:
        # Check file extension
        if not file.filename.endswith(('.csv', '.xlsx', '.xls')):
            raise HTTPException(
                status_code=400, 
                detail="File must be a CSV or Excel file"
            )

        # Create a temporary file to store the uploaded content
        with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(file.filename)[1]) as temp_file:
            content = await file.read()
            temp_file.write(content)
            temp_file_path = temp_file.name

        try:
            # Read the file into a pandas DataFrame
            if file.filename.endswith('.csv'):
                df = pd.read_csv(temp_file_path)
            else:
                df = pd.read_excel(temp_file_path)

            # Initialize ChatOpenAI
            llm = ChatOpenAI(
                model="gpt-3.5-turbo",
                temperature=0
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
            
            # Store original question for response
            original_question = question.strip()
            
            # Format the user question using the template
            formatted_question = USER_PROMPT_TEMPLATE.format(question=original_question)
            
            # Get response from agent using invoke
            response = agent.invoke({"input": formatted_question})
            raw_output = response.get("output", "")

            # Parse the response to extract answer and suggested questions
            answer_parts = raw_output.split("SUGGESTED_QUESTIONS:")
            main_answer = answer_parts[0].replace("ANSWER:", "").strip()
            suggested_questions = []
            
            if len(answer_parts) > 1:
                # Extract questions from numbered list
                questions_text = answer_parts[1].strip()
                for line in questions_text.split("\n"):
                    if line.strip() and any(line.strip().startswith(str(i)) for i in range(1, 4)):
                        question = line.strip().split(".", 1)[1].strip()
                        suggested_questions.append(question)

            execution_time = time.time() - start_time

            return TableQAResponse(
                question=original_question,
                answer=main_answer,
                sources=[file.filename],
                suggested_questions=suggested_questions,
                model_used="GPT-3.5-turbo",
                status_code="200",
                found=True,
                execution_time=round(execution_time, 2)
            )

        finally:
            # Clean up temporary file
            os.unlink(temp_file_path)

    except Exception as e:
        logger.error(f"Error in table_qa: {str(e)}")
        execution_time = time.time() - start_time
        return TableQAResponse(
            question=question,
            answer="",
            sources=[],
            suggested_questions=[],
            model_used="GPT-3.5-turbo",
            status_code="500",
            found=False,
            execution_time=round(execution_time, 2)
        )