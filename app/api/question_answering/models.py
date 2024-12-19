from pydantic import BaseModel
from typing import List, Literal

class QuestionRequest(BaseModel):
    question: str
    max_chunks: int = 10
    model: Literal["gpt-3.5-turbo", "gpt-4", "gpt-4o"] = "gpt-4o"
    num_suggestions: int = 3

class Source(BaseModel):
    filename: str
    document_id: str
    file_type: str
    source_type: str

class QuestionResponse(BaseModel):
    user_id: str
    question: str
    answer: str
    sources: List[Source]
    suggested_questions: List[str]
    model_used: str
    status_code: str
    needs_clarification: bool
    is_followup: bool