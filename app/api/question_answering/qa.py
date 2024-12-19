from fastapi import APIRouter, HTTPException
from fastapi.responses import JSONResponse
import openai
from app.service.log_client import logger
from app.api.question_answering.models import QuestionRequest, QuestionResponse
from app.api.question_answering.memory_buffer import MemoryBuffer
from app.api.question_answering.context_processor import get_context_from_vectors
from app.api.question_answering.answer_generator import process_contexts_in_batches, generate_question_suggestions
from config import OPENAI_API_KEY
from dotenv import load_dotenv
import nltk

load_dotenv()
router = APIRouter()
openai.api_key = OPENAI_API_KEY
nltk.download('wordnet', quiet=True)

memory_buffer = MemoryBuffer()

@router.post("/{user_id}/ask")
async def ask_question(user_id: str, request: QuestionRequest) -> QuestionResponse:
    try:
        if not (1 <= request.max_chunks <= 20):
            raise HTTPException(
                status_code=400,
                detail={
                    "status_code": "400",
                    "error_messages": ["max_chunks must be between 1 and 20"]
                }
            )
            
        if not (1 <= request.num_suggestions <= 10):
            raise HTTPException(
                status_code=400,
                detail={
                    "status_code": "400",
                    "error_messages": ["Number of suggested questions must be between 1 and 10"]
                }
            )

        contexts = await get_context_from_vectors(
            request.question,
            user_id,
            request.max_chunks
        )

        result = await process_contexts_in_batches(
            request.question,
            contexts,
            request.model,
            memory_buffer.get_conversation_context(),
            batch_size=4
        )
        
        await memory_buffer.add_interaction(request.question, result["answer"], contexts)

        suggested_questions = await generate_question_suggestions(
            contexts,
            request.num_suggestions,
            request.model,
            request.question,
            result["needs_clarification"]
        )

        return JSONResponse(
            content={
                "user_id": user_id,
                "question": request.question,
                "answer": result["answer"],
                "sources": [
                    {
                        "filename": src["filename"],
                        "document_id": src["document_id"],
                        "file_type": src["file_type"],
                        "source_type": src["source_type"]
                    }
                    for src in result["sources"]
                ],
                "suggested_questions": suggested_questions,
                "model_used": request.model,
                "status_code": "200",
                "needs_clarification": result["needs_clarification"],
                "is_followup": result["is_followup"]
            }
        )

    except Exception as e:
        logger.error(f"Error processing question | user_id={user_id}, error={str(e)}")
        raise HTTPException(
            status_code=500,
            detail={
                "status_code": "500",
                "error_messages": [str(e)]
            }
        )