from fastapi import APIRouter, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from typing import List, Optional, Literal
import openai
from app.service.openai_client import get_embeddings
from app.service.log_client import logger
from config import PINECONE_CLIENT_INDEX, PINECONE_ANC_INDEX, OPENAI_API_KEY, PINECONE_API_KEY
from pinecone import Pinecone
import os
from dotenv import load_dotenv
from app.variables.keywords import keywords
from nltk import WordNetLemmatizer
import nltk
from datetime import datetime, timezone
from collections import deque

load_dotenv()
router = APIRouter()
openai.api_key = OPENAI_API_KEY
nltk.download('wordnet', quiet=True)
lemmatizer = WordNetLemmatizer()

class MemoryBuffer:
    def __init__(self, buffer_size=3):
        self.conversation_history = deque(maxlen=buffer_size)
        
    def get_context(self):
        messages = []
        for item in self.conversation_history:
            messages.extend([
                {"role": "user", "content": item["question"]},
                {"role": "assistant", "content": item["answer"]}
            ])
        return messages

    def add_interaction(self, question: str, answer: str):
        self.conversation_history.append({
            "question": question,
            "answer": answer
        })

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

memory_buffer = MemoryBuffer()

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
        
        return questions[:n_suggestions]
    except Exception as e:
        logger.error(f"Question suggestion generation failed | error={str(e)}")
        raise

def get_user_namespaces(user_id: str) -> List[str]:
    try:
        pc = Pinecone(api_key=PINECONE_API_KEY)
        index = pc.Index(PINECONE_CLIENT_INDEX) 
        stats = index.describe_index_stats()
        user_namespaces = [ns for ns in stats.namespaces.keys() if ns.startswith(f"{user_id}_")]
        return user_namespaces
    except Exception as e:
        logger.error(f"Error getting user namespaces | user_id={user_id}, error={str(e)}")
        raise

async def get_document_contexts(question_embedding: List[float], user_id: str, max_chunks: int) -> List[dict]:
    try:
        user_namespaces = get_user_namespaces(user_id)
        if not user_namespaces:
            return []

        pc = Pinecone(api_key=PINECONE_API_KEY)
        client_index = pc.Index(PINECONE_CLIENT_INDEX)
        
        matches = []
        for namespace in user_namespaces:
            results = client_index.query(
                vector=question_embedding,
                top_k=max_chunks,
                namespace=namespace,
                include_metadata=True
            )
            matches.extend(results.matches)
        
        contexts = []
        for match in matches:
            metadata = match.metadata or {}
            context = {
                "text": metadata.get("text", ""),
                "filename": metadata.get("filename", ""),
                "document_id": metadata.get("document_id", ""),
                "file_type": metadata.get("file_type", ""),
                "namespace": metadata.get("namespace", ""),
                "score": match.score,
                "source_type": "document"
            }
            
            if context["text"].strip():
                contexts.append(context)
                
        return contexts
        
    except Exception as e:
        logger.error(f"Error getting document contexts | user_id={user_id}, error={str(e)}")
        raise

async def get_url_contexts(question_embedding: List[float], max_chunks: int) -> List[dict]:
    try:
        pc = Pinecone(api_key=PINECONE_API_KEY)
        url_index = pc.Index(PINECONE_ANC_INDEX)
        
        stats = url_index.describe_index_stats()
        url_namespaces = list(stats.namespaces.keys())
        
        matches = []
        if url_namespaces:
            for namespace in url_namespaces:
                try:
                    results = url_index.query(
                        vector=question_embedding,
                        top_k=max_chunks,
                        namespace=namespace,
                        include_metadata=True
                    )
                    matches.extend(results.matches)
                except Exception as e:
                    logger.error(f"Error querying URL namespace | namespace={namespace}, error={str(e)}")
                    continue
        else:
            results = url_index.query(
                vector=question_embedding,
                top_k=max_chunks,
                include_metadata=True
            )
            matches.extend(results.matches)
        
        matches.sort(key=lambda x: x.score, reverse=True)
        top_matches = matches[:max_chunks]
        
        contexts = []
        for match in top_matches:
            metadata = match.metadata or {}
            context = {
                "text": metadata.get("text", ""),
                "filename": metadata.get("url", ""),
                "document_id": metadata.get("document_id", metadata.get("url", "")),
                "file_type": "url",
                "namespace": metadata.get("namespace", ""),
                "score": match.score,
                "source_type": "url"
            }
            
            if context["text"].strip():
                contexts.append(context)
        
        return contexts
        
    except Exception as e:
        logger.error(f"Error getting URL contexts | error={str(e)}")
        raise

async def get_context_from_vectors(question: str, user_id: str, max_chunks: int = 10) -> List[dict]:
    try:
        question_embedding = get_embeddings([question])[0]
        
        document_contexts = await get_document_contexts(question_embedding, user_id, max_chunks)
        url_contexts = await get_url_contexts(question_embedding, max_chunks)
        
        all_contexts = document_contexts + url_contexts
        all_contexts.sort(key=lambda x: x["score"], reverse=True)
        
        return all_contexts[:max_chunks]
        
    except Exception as e:
        logger.error(f"Error getting combined contexts | user_id={user_id}, error={str(e)}")
        raise

async def generate_answer(question: str, contexts: List[dict], model: str) -> dict:
    try:
        if not contexts:
            return {
                "answer": "I cannot find any relevant information in the available documents to answer your question.",
                "sources": []
            }

        formatted_contexts = []
        for i, ctx in enumerate(contexts, 1):
            source_type = "URL" if ctx['source_type'] == "url" else "DOCUMENT"
            formatted_contexts.append(
                f"""[CONTENT_{i}]
SOURCE_TYPE: {source_type}
SOURCE: {ctx['filename']}
TEXT: {ctx['text']}
END_CONTENT_{i}"""
            )
        
        context_text = "\n\n".join(formatted_contexts)
        conversation_context = memory_buffer.get_context()

        system_prompt = """You are a helpful AI assistant answering questions based on the provided context.
Follow these rules:
1. Base your answer ONLY on the provided content blocks marked with [CONTENT_X] and previous conversation context
2. If the answer isn't in the context, say "I cannot find the relevant information in the provided documents."
3. Be clear, concise, and accurate
4. After your answer, specify which content blocks you used in this format:
   <SOURCES_USED>
   CONTENT_1: [URL] example.com
   CONTENT_3: [DOCUMENT] filename.pdf
   </SOURCES_USED>"""

        messages = [
            {"role": "system", "content": system_prompt},
            *conversation_context,
            {"role": "user", "content": f"CONTEXT:\n{context_text}\n\nQUESTION: {question}"}
        ]

        response = await openai.ChatCompletion.acreate(
            model=model,
            messages=messages,
            temperature=0.7,
            max_tokens=800
        )

        full_response = response.choices[0].message.content.strip()
        answer_text = full_response
        used_sources = []
        
        if "<SOURCES_USED>" in full_response:
            parts = full_response.split("<SOURCES_USED>")
            answer_text = parts[0].strip()
            
            sources_section = parts[1].split("</SOURCES_USED>")[0].strip()
            source_lines = [line.strip() for line in sources_section.split('\n') if line.strip()]
            
            for line in source_lines:
                if ":" in line:
                    content_num = line.split(":")[0].strip()
                    content_index = int(content_num.replace("CONTENT_", "")) - 1
                    if content_index < len(contexts):
                        ctx = contexts[content_index]
                        source = {
                            "filename": ctx["filename"],
                            "document_id": ctx["document_id"],
                            "file_type": ctx["file_type"]
                        }
                        if source not in used_sources:
                            used_sources.append(source)

        memory_buffer.add_interaction(question, answer_text)

        return {
            "answer": answer_text,
            "sources": used_sources
        }
    except Exception as e:
        logger.error(f"Error generating answer | error={str(e)}")
        raise

async def validate_insurance_question(question: str) -> bool:
    question_words = set(lemmatizer.lemmatize(word.lower()) for word in question.split())
    lemmatized_keywords = set(lemmatizer.lemmatize(keyword.lower()) for keyword in keywords)
    return len(question_words.intersection(lemmatized_keywords)) > 0

@router.post("/{user_id}/ask")
async def ask_question(user_id: str, request: QuestionRequest) -> QuestionResponse:
    start_time = datetime.now(timezone.utc)
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

        result = await generate_answer(
            request.question,
            contexts,
            request.model
        )

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