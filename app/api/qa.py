import logging
from fastapi import APIRouter, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from typing import List, Literal
import openai
from app.service.openai_client import get_embeddings
from app.service.pinecone_client import initialize_pinecone, query_vectors
from app.service.log_client import logger
from app.get_secret_key import get_secret

router = APIRouter()
index = initialize_pinecone()

# Set OpenAI API key
openai.api_key = get_secret("OPENAI_API_KEY")

class QuestionRequest(BaseModel):
    question: str
    max_chunks: int = 5
    model: Literal["gpt-3.5-turbo", "gpt-4o"] = "gpt-3.5-turbo"
    num_suggestions: int = 5

async def get_relevant_chunks(question: str, max_chunks: int) -> List[dict]:
    try:
        question_embedding = get_embeddings([question])[0]
        logger.info(f"Generated embedding for question: {question[:100]}...")

        results = query_vectors(index, question_embedding, top_k=max_chunks)
        logger.info(f"Retrieved {len(results)} relevant chunks")

        chunks = []
        for result in results:
            chunks.append({
                "text": result.metadata.get('text', ''),
                "pdf_id": result.metadata.get('pdf_id', ''),
                "score": result.score
            })
        
        return chunks
    except Exception as e:
        logger.error(f"Error getting relevant chunks: {str(e)}")
        raise

async def generate_answer(question: str, context_chunks: List[dict], model: str) -> str:
    try:
        context = "\n\n".join([chunk["text"] for chunk in context_chunks])
        
        prompt = f"""Based on the following context, answer the question. 
        If the answer cannot be found in the context, say "I cannot find an answer to this question in the provided documents."
        
        Context:
        {context}
        
        Question: {question}
        
        Answer:"""
        
        response = openai.ChatCompletion.create(
            model=model,
            messages=[
                {"role": "system", "content": "You are a helpful assistant that answers questions based on provided context."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.7,
            max_tokens=500
        )
        
        answer = response.choices[0].message.content
        logger.info(f"Generated answer using {model}")
        
        return answer
    except Exception as e:
        logger.error(f"Error generating answer: {str(e)}")
        raise

async def generate_question_suggestions(context_chunks: List[dict], n_suggestions: int, model: str) -> List[str]:
    try:
        context = "\n\n".join([chunk["text"] for chunk in context_chunks])
        
        prompt = f"""Based on the following text, generate exactly {n_suggestions} relevant and specific questions that can be answered using this content. 
        The questions should be:
        1. Diverse and cover different aspects of the content
        2. Specific enough to be answered from the given context
        3. Focused on important information in the text
        
        Format: Number each question (1., 2., etc.)
        
        Text:
        {context}
        
        Generate exactly {n_suggestions} questions:"""
        
        response = openai.ChatCompletion.create(
            model=model,
            messages=[
                {"role": "system", "content": "You are a helpful assistant that generates relevant questions based on provided content."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.8,
            max_tokens=300
        )
        
        # Parse the response to extract questions
        suggested_questions = []
        for line in response.choices[0].message.content.strip().split('\n'):
            if line.strip() and any(line.strip().startswith(f"{i}.") for i in range(1, n_suggestions + 1)):
                question = line.split('.', 1)[1].strip()
                suggested_questions.append(question)
        
        logger.info(f"Generated {len(suggested_questions)} question suggestions using {model}")
        return suggested_questions
    except Exception as e:
        logger.error(f"Error generating question suggestions: {str(e)}")
        raise

@router.post("/ask")
async def ask_question(request: QuestionRequest):
    try:
        logger.info(f"Received question: {request.question} (using model: {request.model})")
        
        # Validate num_suggestions
        if request.num_suggestions < 1 or request.num_suggestions > 10:
            raise HTTPException(
                status_code=400,
                detail="Number of suggested questions must be between 1 and 10"
            )
        
        # Get relevant chunks
        relevant_chunks = await get_relevant_chunks(request.question, request.max_chunks)
        
        if not relevant_chunks:
            return JSONResponse(content={
                "answer": "No relevant information found in the documents.",
                "chunks": [],
                "suggested_questions": [],
                "model_used": request.model
            })
        
        # Generate answer and question suggestions concurrently
        answer = await generate_answer(request.question, relevant_chunks, request.model)
        suggested_questions = await generate_question_suggestions(
            relevant_chunks, 
            request.num_suggestions,
            request.model
        )
        
        return JSONResponse(content={
            "question": request.question,
            "answer": answer,
            # "chunks": relevant_chunks,
            "suggested_questions": suggested_questions,
            "model_used": request.model
        })
    
    except Exception as e:
        logger.error(f"Error processing question: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))