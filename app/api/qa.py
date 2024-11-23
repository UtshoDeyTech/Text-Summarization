from fastapi import APIRouter, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from typing import List, Optional, Literal
import openai
from app.service.openai_client import get_embeddings
from app.service.log_client import logger
from app.get_secret_key import get_secret
from pinecone import Pinecone

router = APIRouter()
openai.api_key = get_secret("OPENAI_API_KEY")

class QuestionRequest(BaseModel):
    question: str
    max_chunks: int = 5
    model: Literal["gpt-3.5-turbo", "gpt-4", "gpt-4o"] = "gpt-4o"
    num_suggestions: int = 5

class Source(BaseModel):
    filename: str
    document_id: str
    file_type: str

class QuestionResponse(BaseModel):
    user_id: str
    question: str
    answer: str
    sources: List[Source]
    suggested_questions: List[str]
    model_used: str
    status_code: str

async def generate_question_suggestions(context_chunks: List[dict], n_suggestions: int, model: str, original_question: str) -> List[str]:
    try:
        if not context_chunks:
            return []
            
        formatted_contexts = [
            f"""Content: {chunk["text"]}
Source: {chunk["filename"]}
---""" for chunk in context_chunks[:5]  # Limit to first 5 chunks for suggestions
        ]
        
        context = "\n".join(formatted_contexts)
        prompt = f"""Based on ONLY the provided content, generate {n_suggestions} questions.

Content:
{context}

Rules:
1. Questions must be answerable using ONLY the provided content
2. Questions should be different from: "{original_question}"
3. Format as numbered list (e.g. 1., 2., etc)
4. Questions should be relevant and meaningful
5. Questions should explore different aspects of the content"""
        
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
        
        logger.info(f"Generated suggested questions | count={len(questions)}")
        return questions[:n_suggestions]
    except Exception as e:
        logger.error(f"Question suggestion generation failed | error={str(e)}")
        raise

def get_user_namespaces(user_id: str) -> List[str]:
    """Get all namespaces for a specific user."""
    try:
        pc = Pinecone(api_key=get_secret("PINECONE_API_KEY"))
        index = pc.Index("client-document")
        
        stats = index.describe_index_stats()
        user_namespaces = [ns for ns in stats.namespaces.keys() if ns.startswith(f"{user_id}_")]
        
        logger.info(f"Found namespaces | user_id={user_id}, namespace_count={len(user_namespaces)}")
        return user_namespaces
    except Exception as e:
        logger.error(f"Error getting user namespaces | user_id={user_id}, error={str(e)}")
        raise

async def get_context_from_vectors(question: str, user_id: str, max_chunks: int = 10) -> List[dict]:
    try:
        # Get embeddings for the question
        question_embedding = get_embeddings([question])[0]
        
        # Get all namespaces for this user
        user_namespaces = get_user_namespaces(user_id)
        
        if not user_namespaces:
            logger.warning(f"No namespaces found | user_id={user_id}")
            return []

        # Initialize Pinecone client
        pc = Pinecone(api_key=get_secret("PINECONE_API_KEY"))
        index = pc.Index("client-document")
        
        # Query each namespace and collect results
        all_matches = []
        for namespace in user_namespaces:
            results = index.query(
                vector=question_embedding,
                top_k=max_chunks,
                namespace=namespace,
                include_metadata=True
            )
            all_matches.extend(results.matches)
        
        # Sort all matches by score and take top max_chunks
        all_matches.sort(key=lambda x: x.score, reverse=True)
        top_matches = all_matches[:max_chunks]
        
        # Format the results
        contexts = []
        for match in top_matches:
            metadata = match.metadata or {}
            context = {
                "text": metadata.get("text", ""),
                "filename": metadata.get("filename", ""),
                "document_id": metadata.get("document_id", ""),
                "file_type": metadata.get("file_type", ""),
                "namespace": metadata.get("namespace", ""),
                "score": match.score
            }
            
            if context["text"].strip():
                contexts.append(context)
        
        logger.info(f"Retrieved contexts | user_id={user_id}, contexts_found={len(contexts)}")
        return contexts
    except Exception as e:
        logger.error(f"Error getting context | user_id={user_id}, error={str(e)}")
        raise

async def generate_answer(question: str, contexts: List[dict], model: str) -> dict:
    try:
        if not contexts:
            return {
                "answer": "I cannot find any relevant information in the available documents to answer your question.",
                "sources": []
            }

        # Format context for the prompt
        formatted_contexts = []
        for i, ctx in enumerate(contexts, 1):
            formatted_contexts.append(
                f"""[CONTENT_{i}]
SOURCE: {ctx['filename']}
TEXT: {ctx['text']}
END_CONTENT_{i}"""
            )
        
        context_text = "\n\n".join(formatted_contexts)

        system_prompt = """You are a helpful AI assistant answering questions based on the provided context.
Follow these rules:
1. Base your answer ONLY on the provided content blocks marked with [CONTENT_X]
2. If the answer isn't in the context, say "I cannot find the relevant information in the provided documents"
3. Be clear, concise, and accurate
4. After your answer, you must specify which content blocks you used in this format:
   <SOURCES_USED>
   CONTENT_1: filename1.pdf
   CONTENT_3: filename2.docx
   </SOURCES_USED>
5. Only include content blocks that directly contributed to your answer
6. Do not mention content block numbers in your answer text"""

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": f"CONTEXT:\n{context_text}\n\nQUESTION: {question}"}
        ]

        response = await openai.ChatCompletion.acreate(
            model=model,
            messages=messages,
            temperature=0.7,
            max_tokens=800
        )

        full_response = response.choices[0].message.content.strip()
        
        # Split response into answer and sources
        answer_text = full_response
        used_sources = []
        
        if "<SOURCES_USED>" in full_response:
            parts = full_response.split("<SOURCES_USED>")
            answer_text = parts[0].strip()
            
            # Extract source information
            sources_section = parts[1].split("</SOURCES_USED>")[0].strip()
            source_lines = [line.strip() for line in sources_section.split('\n') if line.strip()]
            
            # Map content numbers to actual contexts
            for line in source_lines:
                if ":" in line:
                    content_num, _ = line.split(":", 1)
                    content_index = int(content_num.replace("CONTENT_", "")) - 1
                    
                    if content_index < len(contexts):
                        ctx = contexts[content_index]
                        source = {
                            "filename": ctx["filename"],
                            "document_id": ctx["document_id"],
                            "file_type": ctx["file_type"]
                        }
                        if source not in used_sources:  # Avoid duplicates
                            used_sources.append(source)

        # If no sources were specified but we got an answer, include all sources
        if not used_sources and "cannot find" not in answer_text.lower():
            seen_docs = set()
            for ctx in contexts:
                doc_key = (ctx["document_id"], ctx["filename"])
                if doc_key not in seen_docs:
                    seen_docs.add(doc_key)
                    used_sources.append({
                        "filename": ctx["filename"],
                        "document_id": ctx["document_id"],
                        "file_type": ctx["file_type"]
                    })

        return {
            "answer": answer_text,
            "sources": used_sources
        }
    except Exception as e:
        logger.error(f"Error generating answer | error={str(e)}")
        raise

@router.post("/{user_id}/ask")
async def ask_question(user_id: str, request: QuestionRequest) -> QuestionResponse:
    try:
        # Input validation
        if request.max_chunks < 1 or request.max_chunks > 20:
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

        # Get relevant context from vector store
        contexts = await get_context_from_vectors(
            request.question,
            user_id,
            request.max_chunks
        )

        # Generate answer using OpenAI
        result = await generate_answer(
            request.question,
            contexts,
            request.model
        )

        # Generate suggested questions
        suggested_questions = await generate_question_suggestions(
            contexts,
            request.num_suggestions,
            request.model,
            request.question
        )

        return JSONResponse(
            content={
                "user_id": user_id,
                "question": request.question,
                "answer": result["answer"],
                "sources": result["sources"],
                "suggested_questions": suggested_questions,
                "model_used": request.model,
                "status_code": "200"
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