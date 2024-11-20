import logging
from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from typing import List, Literal, Dict
import openai
from app.service.openai_client import get_embeddings
from app.service.pinecone_client import query_vectors
from app.service.log_client import logger
from app.get_secret_key import get_secret

router = APIRouter()
openai.api_key = get_secret("OPENAI_API_KEY")

class QuestionRequest(BaseModel):
    question: str
    max_chunks: int = 10
    model: Literal["gpt-3.5-turbo", "gpt-4", "gpt-4o"] = "gpt-4o"
    num_suggestions: int = 5

async def get_relevant_chunks(question: str, max_chunks: int, user_id: str) -> List[dict]:
    try:
        question_embedding = get_embeddings([question])[0]
        results = query_vectors(user_id, question_embedding, top_k=max_chunks)
        
        chunks = [{
            "text": result.metadata.get('text', ''),
            "document_id": result.metadata.get('document_id', ''),
            "filename": result.metadata.get('filename', ''),
            "file_type": result.metadata.get('file_type', ''),
            "score": result.score
        } for result in results]
        
        logger.info(f"Retrieved chunks | user_id={user_id}, chunks_found={len(chunks)}")
        return chunks
    except Exception as e:
        logger.error(f"Chunk retrieval failed | user_id={user_id}, error={str(e)}")
        raise

async def direct_openai_query(question: str, model: str) -> str:
    try:
        response = await openai.ChatCompletion.acreate(
            model=model,
            messages=[
                {"role": "system", "content": "Provide concise answers (max 200 words). If you don't know, say 'I don't know'."},
                {"role": "user", "content": question}
            ],
            temperature=0.7,
            max_tokens=300
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        logger.error(f"Direct OpenAI query failed | error={str(e)}")
        raise

async def generate_answer(question: str, context_chunks: List[dict], model: str) -> Dict:
    try:
        formatted_contexts = []
        chunk_map = {}
        for i, chunk in enumerate(context_chunks):
            formatted_contexts.append(f"""
Content [{i+1}]: {chunk["text"]}
Source Document: {chunk["filename"]} (ID: {chunk["document_id"]}, Type: {chunk["file_type"]})
---""")
            chunk_map[i+1] = {
                "filename": chunk["filename"], 
                "document_id": chunk["document_id"],
                "file_type": chunk["file_type"]
            }
        
        context = "\n".join(formatted_contexts)
        
        system_prompt = """You are a helpful assistant that answers questions based on provided context.
Your answers should be based solely on the provided context. If the answer cannot be found in the context,
say "I cannot find an answer to this question in the provided documents."

IMPORTANT: 
1. Do not include source information or citations in your answer text. 
2. After formulating your answer, specify which Content blocks you used in a JSON structure.
3. Format: {"used_content_blocks": [1, 3]} (if you used content from blocks 1 and 3)
4. Separate answer and JSON with three hyphens (---)
5. If no answer found, don't include content blocks."""

        response = await openai.ChatCompletion.acreate(
            model=model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": f"Context Information:\n{context}\n\nQuestion: {question}"}
            ],
            temperature=0.7,
            max_tokens=800
        )
        
        parts = response.choices[0].message.content.strip().split('---')
        answer = parts[0].strip()
        
        if "I cannot find an answer" in answer:
            return {"answer": answer, "sources": []}
        
        used_blocks = []
        if len(parts) > 1:
            try:
                import json
                blocks_info = json.loads(parts[1].strip())
                used_blocks = blocks_info.get('used_content_blocks', [])
            except:
                used_blocks = list(range(1, len(context_chunks) + 1))
        else:
            used_blocks = list(range(1, len(context_chunks) + 1))
            
        sources = [{
            "filename": chunk_map[block]["filename"],
            "document_id": chunk_map[block]["document_id"],
            "file_type": chunk_map[block]["file_type"]
        } for block in used_blocks if block in chunk_map]
        
        return {"answer": answer, "sources": sources}
    except Exception as e:
        logger.error(f"Answer generation failed | error={str(e)}")
        raise

async def process_chunks_in_batches(chunks: List[dict], question: str, model: str, batch_size: int = 5) -> Dict:
    try:
        all_answers = []
        all_sources = []
        
        for i in range(0, len(chunks), batch_size):
            batch = chunks[i:i + batch_size]
            result = await generate_answer(question, batch, model)
            
            if result["answer"] and "cannot find an answer" not in result["answer"].lower():
                all_answers.append(result["answer"])
                all_sources.extend(result["sources"])
        
        if not all_answers:
            direct_answer = await direct_openai_query(question, model)
            return {
                "answer": direct_answer if "I don't know" not in direct_answer else "I cannot find an answer to this question.",
                "sources": []
            }
        
        combine_prompt = f"""Combine these answers into a single coherent response:

Answers:
{chr(10).join(f'{i+1}. {answer}' for i, answer in enumerate(all_answers))}

Requirements:
1. Synthesize the information into one comprehensive answer
2. Remove any redundancy
3. Maintain accuracy
4. Make the answer flow naturally"""

        response = await openai.ChatCompletion.acreate(
            model=model,
            messages=[
                {"role": "system", "content": "Combine multiple answers into one coherent response."},
                {"role": "user", "content": combine_prompt}
            ],
            temperature=0.7,
            max_tokens=800
        )
        
        final_answer = response.choices[0].message.content.strip()
        
        unique_sources = []
        seen = set()
        for source in all_sources:
            key = (source["document_id"], source["filename"])
            if key not in seen:
                seen.add(key)
                unique_sources.append(source)
        
        return {
            "answer": final_answer,
            "sources": unique_sources
        }
    except Exception as e:
        logger.error(f"Batch processing failed | error={str(e)}")
        raise

async def generate_question_suggestions(context_chunks: List[dict], n_suggestions: int, model: str, original_question: str) -> List[str]:
    try:
        if not context_chunks:
            return []
            
        formatted_contexts = [
            f"""Content: {chunk["text"]}
Source: {chunk["filename"]}
---""" for chunk in context_chunks[:5]
        ]
        
        context = "\n".join(formatted_contexts)
        prompt = f"""Based on ONLY the provided content, generate {n_suggestions} questions.

Content:
{context}

Rules:
1. Questions must be answerable using ONLY the provided content
2. Questions should be different from: "{original_question}"
3. Format as numbered list (e.g. 1., 2., etc)"""
        
        response = await openai.ChatCompletion.acreate(
            model=model,
            messages=[
                {"role": "system", "content": "Generate questions answerable only from the provided content."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.8,
            max_tokens=300
        )
        
        questions = [
            line.split('.', 1)[1].strip()
            for line in response.choices[0].message.content.strip().split('\n')
            if line.strip() and any(line.strip().startswith(f"{i}.") for i in range(1, n_suggestions + 1))
        ]
        
        return questions[:n_suggestions]
    except Exception as e:
        logger.error(f"Question suggestion generation failed | error={str(e)}")
        raise

@router.post("/{user_id}/ask")
async def ask_question(request: Request, user_id: str, question_request: QuestionRequest):
    try:
        if not (1 <= question_request.num_suggestions <= 10):
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
        
        suggested_questions = []
        if relevant_chunks:
            result = await process_chunks_in_batches(
                relevant_chunks,
                question_request.question,
                question_request.model
            )
            suggested_questions = await generate_question_suggestions(
                relevant_chunks,
                question_request.num_suggestions,
                question_request.model,
                question_request.question
            )
        else:
            direct_answer = await direct_openai_query(
                question_request.question,
                question_request.model
            )
            result = {
                "answer": direct_answer if "I don't know" not in direct_answer else "No relevant information found.",
                "sources": []
            }
        
        return JSONResponse(content={
            "user_id": user_id,
            "question": question_request.question,
            "answer": result["answer"],
            "sources": result["sources"],
            "suggested_questions": suggested_questions,
            "model_used": question_request.model,
            "status_code": "200"
        })
        
    except Exception as e:
        logger.error(f"Question processing failed | user_id={user_id}, error={str(e)}")
        raise HTTPException(
            status_code=500,
            detail={
                "status_code": "500",
                "error_messages": [str(e)]
            }
        )