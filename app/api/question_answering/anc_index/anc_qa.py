from fastapi import APIRouter, HTTPException
from typing import List
from pydantic import BaseModel, Field
from openai import AsyncOpenAI
import json
import time
from pinecone import Pinecone
from app.service.log_client import logger
from app.service.openai_client import get_embeddings
from config import OPENAI_API_KEY, PINECONE_ANC_INDEX, PINECONE_API_KEY, MODEL, MAX_CHUNKS, NUM_SUGGESTIONS

router = APIRouter()
client = AsyncOpenAI(api_key=OPENAI_API_KEY)

class QuestionRequest(BaseModel):
    question: str = Field(..., description="The question to be answered")

class Source(BaseModel):
    document_id: str
    content_type: str
    url: str = ""
    filename: str = ""
    form_url: str = ""
    upload_date: str = ""

class QuestionResponse(BaseModel):
    question: str
    answer: str
    sources: List[Source] = []
    suggested_questions: List[str] = []
    model_used: str
    status_code: str = "200"
    found: bool
    execution_time: float

def get_system_prompt() -> str:
    return f"""You are an assistant analyzing provided contexts to answer questions. Always generate {NUM_SUGGESTIONS} relevant follow-up questions.

Your response MUST be a valid JSON object with EXACTLY these fields:
{{
    "answer": "Answer based on context information",
    "found": true/false,
    "source": {{
        "document_id": "ID from metadata",
        "content_type": "url" or "document",
        "upload_date": "ISO date from metadata",
        "url": "URL if content_type is url",
        "filename": "filename if content_type is document",
        "form_url": "form URL if content_type is document"
    }},
    "suggested_questions": [
        "Specific follow-up question 1",
        "Specific follow-up question 2",
        "Specific follow-up question 3"
    ]
}}

Rules:
1. source fields depend on content_type - include ALL metadata from the chunk
2. set found=true if ANY relevant information exists
3. include partial answers if available
4. preserve original metadata structure and dates"""

def create_source_from_metadata(metadata: dict) -> Source:
    return Source(
        document_id=metadata.get("document_id", ""),
        content_type=metadata.get("content_type", ""),
        url=metadata.get("url", ""),
        filename=metadata.get("filename", ""),
        form_url=metadata.get("form_url", ""),
        upload_date=metadata.get("upload_date", "")
    )

@router.post("/ask-anc", response_model=QuestionResponse)
async def ask_question(request: QuestionRequest):
    start_time = time.time()
    logger.info(f"Processing question | question={request.question}")
    
    try:
        # Get embeddings asynchronously
        embeddings = await get_embeddings([request.question])
        question_embedding = embeddings[0]
        
        pc = Pinecone(api_key=PINECONE_API_KEY)
        index = pc.Index(PINECONE_ANC_INDEX)
        
        stats = index.describe_index_stats()
        namespaces = list(stats.namespaces.keys() if stats.namespaces else [])
        
        # Collect matches from all namespaces with score threshold
        all_matches = []
        for namespace in namespaces:
            try:
                results = index.query(
                    vector=question_embedding,
                    top_k=3,
                    include_metadata=True,
                    namespace=namespace,
                    score_threshold=0.5  # Only return matches with score > 0.5
                )
                all_matches.extend(results.matches)
            except Exception as e:
                logger.error(f"Namespace query error | namespace={namespace}, error={str(e)}")
                continue
        
        # Sort and take top max_chunks
        top_matches = []
        if all_matches:
            all_matches.sort(key=lambda x: x.score, reverse=True)
            top_matches = all_matches[:MAX_CHUNKS]

        if not top_matches:
            return QuestionResponse(
                question=request.question,
                answer="No relevant information found.",
                sources=[],
                suggested_questions=[],
                model_used=MODEL,
                status_code="200",
                found=False,
                execution_time=round(time.time() - start_time, 2)
            )

        # Combine context from top matches
        contexts = []
        for match in top_matches:
            metadata = match.metadata or {}
            text = metadata.get("text", "")  # Use full chunk
            contexts.append(f"Context from {metadata.get('document_id')} (similarity: {match.score:.2f}):\n{text}")
        
        combined_context = "\n\n".join(contexts)
        source = create_source_from_metadata(top_matches[0].metadata)
        
        # Use async OpenAI client
        response = await client.chat.completions.create(
            model=MODEL,
            messages=[
                {"role": "system", "content": get_system_prompt()},
                {"role": "user", "content": f"Question: {request.question}\n\n{combined_context}"}
            ],
            temperature=0.7,
            max_tokens=300,
            response_format={"type": "json_object"}
        )

        try:
            raw_response = response.choices[0].message.content
            # logger.info(f"OpenAI raw response: {raw_response}")
            gpt_response = json.loads(raw_response)
            execution_time = round(time.time() - start_time, 2)
            
            return QuestionResponse(
                question=request.question,
                answer=gpt_response.get("answer", ""),
                sources=[source],
                suggested_questions=gpt_response.get("suggested_questions", []),
                model_used=MODEL,
                status_code="200",
                found=gpt_response.get("found", False),
                execution_time=execution_time
            )
            
        except json.JSONDecodeError as e:
            logger.error(f"OpenAI response parse error | error={str(e)}")
            return QuestionResponse(
                question=request.question,
                answer="Error processing response.",
                sources=[],
                suggested_questions=[],
                model_used=MODEL,
                status_code="500",
                found=False,
                execution_time=round(time.time() - start_time, 2)
            )

    except Exception as e:
        logger.error(f"Question processing error | error={str(e)}")
        raise HTTPException(
            status_code=500,
            detail={
                "status_code": "500",
                "error_messages": [str(e)],
                "execution_time": round(time.time() - start_time, 2)
            }
        )