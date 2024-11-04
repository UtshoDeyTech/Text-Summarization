import logging
from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from typing import List, Literal
import openai
from app.service.openai_client import get_embeddings
from app.service.pinecone_client import query_vectors
from app.service.log_client import logger
from app.get_secret_key import get_secret

router = APIRouter()
openai.api_key = get_secret("OPENAI_API_KEY")

class QuestionRequest(BaseModel):
    question: str
    max_chunks: int = 5
    model: Literal["gpt-3.5-turbo", "gpt-4"] = "gpt-3.5-turbo"
    num_suggestions: int = 5

async def get_relevant_chunks(question: str, max_chunks: int, user_id: str) -> List[dict]:
    try:
        logger.info(f"Getting relevant chunks | user_id={user_id}, question_length={len(question)}, max_chunks={max_chunks}")
        question_embedding = get_embeddings([question])[0]
        results = query_vectors(user_id, question_embedding, top_k=max_chunks)
        
        chunks = [{
            "text": result.metadata.get('text', ''),
            "pdf_id": result.metadata.get('pdf_id', ''),
            "filename": result.metadata.get('filename', ''),
            "score": result.score
        } for result in results]
        
        logger.info(f"Retrieved chunks | user_id={user_id}, chunks_found={len(chunks)}")
        return chunks
    except Exception as e:
        logger.error(f"Chunk retrieval failed | user_id={user_id}, error_type={type(e).__name__}, error={str(e)}")
        raise

async def generate_answer(question: str, context_chunks: List[dict], model: str) -> str:
    try:
        logger.info(f"Generating answer | model={model}, context_chunks={len(context_chunks)}")
        context = "\n\n".join([chunk["text"] for chunk in context_chunks])
        prompt = f"""Based on the following context, answer the question. 
        If the answer cannot be found in the context, say "I cannot find an answer to this question in the provided documents."
        
        Context:
        {context}
        
        Question: {question}
        
        Answer:"""
        
        response = await openai.ChatCompletion.acreate(
            model=model,
            messages=[
                {"role": "system", "content": "You are a helpful assistant that answers questions based on provided context."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.7,
            max_tokens=500
        )
        
        answer = response.choices[0].message.content
        logger.info(f"Answer generation successful | model={model}, answer_length={len(answer)}")
        return answer
    except Exception as e:
        logger.error(f"Answer generation failed | model={model}, error_type={type(e).__name__}, error={str(e)}")
        raise

async def generate_question_suggestions(context_chunks: List[dict], n_suggestions: int, model: str) -> List[str]:
    try:
        logger.info(f"Generating question suggestions | model={model}, n_suggestions={n_suggestions}")
        context = "\n\n".join([chunk["text"] for chunk in context_chunks])
        prompt = f"""Based on the following text, generate exactly {n_suggestions} relevant questions that can be answered using this content.
        Format: Number each question (1., 2., etc.)
        
        Text:
        {context}
        
        Generate {n_suggestions} questions:"""
        
        response = await openai.ChatCompletion.acreate(
            model=model,
            messages=[
                {"role": "system", "content": "You are a helpful assistant that generates relevant questions based on provided content."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.8,
            max_tokens=300
        )
        
        suggested_questions = []
        for line in response.choices[0].message.content.strip().split('\n'):
            if line.strip() and any(line.strip().startswith(f"{i}.") for i in range(1, n_suggestions + 1)):
                question = line.split('.', 1)[1].strip()
                suggested_questions.append(question)
        
        logger.info(f"Question suggestions generated | count={len(suggested_questions)}")
        return suggested_questions
    except Exception as e:
        logger.error(f"Question suggestion generation failed | model={model}, error_type={type(e).__name__}, error={str(e)}")
        raise

@router.post("/{user_id}/ask")
async def ask_question(
    request: Request,
    user_id: str,
    question_request: QuestionRequest
):
    try:
        logger.info(
            f"Processing question request | "
            f"user_id={user_id}, "
            f"model={question_request.model}, "
            f"max_chunks={question_request.max_chunks}, "
            f"num_suggestions={question_request.num_suggestions}"
        )
        
        if not (1 <= question_request.num_suggestions <= 10):
            logger.warning(f"Invalid suggestion count | user_id={user_id}, num_suggestions={question_request.num_suggestions}")
            raise HTTPException(
                status_code=400,
                detail={
                    "status_code": "400",
                    "error_messages": ["Number of suggested questions must be between 1 and 10"]
                }
            )
        
        relevant_chunks = await get_relevant_chunks(
            question_request.question, 
            question_request.max_chunks,
            user_id
        )
        
        if not relevant_chunks:
            logger.info(f"No relevant chunks found | user_id={user_id}")
            return JSONResponse(content={
                "user_id": user_id,
                "answer": "No relevant information found in your documents.",
                "suggested_questions": [],
                "model_used": question_request.model,
                "status_code": "200"
            })
        
        answer = await generate_answer(
            question_request.question, 
            relevant_chunks, 
            question_request.model
        )
        
        suggested_questions = await generate_question_suggestions(
            relevant_chunks, 
            question_request.num_suggestions,
            question_request.model
        )
        
        logger.info(
            f"Question processing successful | "
            f"user_id={user_id}, "
            f"answer_length={len(answer)}, "
            f"suggestions_count={len(suggested_questions)}"
        )
        
        return JSONResponse(content={
            "user_id": user_id,
            "question": question_request.question,
            "answer": answer,
            "suggested_questions": suggested_questions,
            "model_used": question_request.model,
            "status_code": "200"
        })
        
    except Exception as e:
        error_msg = (
            f"Question processing failed | "
            f"user_id={user_id}, "
            f"error_type={type(e).__name__}, "
            f"error={str(e)}"
        )
        logger.error(error_msg)
        raise HTTPException(
            status_code=500,
            detail={
                "status_code": "500",
                "error_messages": [str(e)]
            }
        )