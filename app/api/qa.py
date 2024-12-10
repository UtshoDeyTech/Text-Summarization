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
        self.context_history = deque(maxlen=buffer_size)
        
    async def generate_context_summary(self, contexts: List[dict]) -> str:
        context_text = "\n".join([
            f"Source: {ctx['filename']} ({ctx['source_type']})\nContent: {ctx['text']}"
            for ctx in contexts
        ])
        
        response = await openai.ChatCompletion.acreate(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "Create a brief summary of the key information from these documents."},
                {"role": "user", "content": context_text}
            ],
            temperature=0.7,
            max_tokens=100
        )
        return response.choices[0].message.content

    async def add_interaction(self, question: str, answer: str, contexts: List[dict]):
        summary = await self.generate_context_summary(contexts)
        self.conversation_history.append({
            "question": question,
            "answer": answer,
            "timestamp": datetime.now(timezone.utc),
            "context_summary": summary,
            "sources": [{"filename": ctx["filename"], "source_type": ctx["source_type"]} for ctx in contexts]
        })
        self.context_history.append(contexts)

    def get_conversation_context(self):
        messages = []
        for item in self.conversation_history:
            messages.extend([
                {"role": "user", "content": item["question"]},
                {"role": "assistant", "content": f"{item['answer']}\nContext: {item['context_summary']}"}
            ])
        return messages

    def is_followup_question(self, current_question: str) -> tuple[bool, List[dict]]:
        if not self.conversation_history:
            return False, []
            
        # Check if current question is related to previous context
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "Determine if the new question is a follow-up to the previous conversation."},
                {"role": "user", "content": f"Previous conversation:\n{self.conversation_history[-1]['question']}\n{self.conversation_history[-1]['answer']}\n\nNew question: {current_question}"}
            ],
            temperature=0.3,
            max_tokens=50
        )
        
        is_followup = "yes" in response.choices[0].message.content.lower()
        relevant_contexts = self.context_history[-1] if is_followup else []
        
        return is_followup, relevant_contexts

class QuestionRequest(BaseModel):
    question: str
    max_chunks: int = 5
    model: Literal["gpt-3.5-turbo", "gpt-4", "gpt-4o"] = "gpt-4o"
    num_suggestions: int = 5

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

memory_buffer = MemoryBuffer()

async def get_context_from_vectors(question: str, user_id: str, max_chunks: int = 10) -> List[dict]:
    try:
        question_embedding = get_embeddings([question])[0]
        
        # Get client contexts
        pc = Pinecone(api_key=PINECONE_API_KEY)
        client_index = pc.Index(PINECONE_CLIENT_INDEX)
        
        client_namespaces = [ns for ns in client_index.describe_index_stats().namespaces.keys() 
                            if ns.startswith(f"{user_id}_")]
        
        client_contexts = []
        for namespace in client_namespaces:
            results = client_index.query(
                vector=question_embedding,
                top_k=max_chunks,
                namespace=namespace,
                include_metadata=True
            )
            for match in results.matches:
                metadata = match.metadata or {}
                context = {
                    "text": metadata.get("text", ""),
                    "filename": metadata.get("filename", ""),
                    "document_id": metadata.get("document_id", ""),
                    "file_type": metadata.get("file_type", ""),
                    "source_type": "client",
                    "namespace": namespace,
                    "score": match.score
                }
                if context["text"].strip():
                    client_contexts.append(context)

        # Get global contexts
        global_index = pc.Index(PINECONE_ANC_INDEX)
        global_results = global_index.query(
            vector=question_embedding,
            top_k=max_chunks,
            include_metadata=True
        )
        
        global_contexts = []
        for match in global_results.matches:
            metadata = match.metadata or {}
            context = {
                "text": metadata.get("text", ""),
                "filename": metadata.get("url", ""),
                "document_id": metadata.get("document_id", ""),
                "file_type": "url",
                "source_type": "global",
                "score": match.score
            }
            if context["text"].strip():
                global_contexts.append(context)

        # Combine and sort contexts, prioritizing client contexts
        all_contexts = sorted(client_contexts + global_contexts, 
                            key=lambda x: (x["source_type"] != "client", -x["score"]))
                            
        return all_contexts[:max_chunks]
        
    except Exception as e:
        logger.error(f"Error getting contexts | user_id={user_id}, error={str(e)}")
        raise

async def generate_answer(question: str, contexts: List[dict], model: str) -> dict:
    try:
        if not contexts:
            return {
                "answer": "I cannot find relevant information to answer your question.",
                "sources": [],
                "needs_clarification": False
            }

        # Check for multiple client sources
        client_sources = {
            ctx["filename"] for ctx in contexts 
            if ctx["source_type"] == "client"
        }
        needs_clarification = len(client_sources) > 1

        # Format contexts
        formatted_contexts = []
        for i, ctx in enumerate(contexts, 1):
            formatted_contexts.append(
                f"""[CONTENT_{i}]
SOURCE_TYPE: {ctx['source_type'].upper()}
SOURCE: {ctx['filename']}
TEXT: {ctx['text']}
END_CONTENT_{i}"""
            )
        
        context_text = "\n\n".join(formatted_contexts)
        conversation_context = memory_buffer.get_conversation_context()

        # Check if this is a follow-up question
        is_followup, previous_contexts = memory_buffer.is_followup_question(question)
        
        system_prompt = f"""You are a helpful AI assistant answering questions based on the provided context.
Rules:
1. Base answers ONLY on the provided content blocks and conversation history
2. If the answer isn't in the context, say so
3. Prioritize information from CLIENT sources over GLOBAL sources
4. {'Since this appears to be a follow-up question, consider the previous context in your answer.' if is_followup else ''}
5. {'Specify the exact client or data source you want to query (like client ID, name, or date range).' if needs_clarification else ''}

After your answer, list the sources used in this format:
<SOURCES_USED>
CONTENT_1: [CLIENT] filename.pdf
CONTENT_3: [GLOBAL] example.com
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

        answer = response.choices[0].message.content
        
        # Extract sources used
        sources = []
        if "<SOURCES_USED>" in answer:
            answer_parts = answer.split("<SOURCES_USED>")
            main_answer = answer_parts[0].strip()
            sources_text = answer_parts[1].split("</SOURCES_USED>")[0].strip()
            
            for line in sources_text.splitlines():
                if ":" in line:
                    content_num = line.split(":")[0].strip()
                    if content_num.startswith("CONTENT_"):
                        idx = int(content_num.replace("CONTENT_", "")) - 1
                        if idx < len(contexts):
                            sources.append(contexts[idx])
        else:
            main_answer = answer

        return {
            "answer": main_answer,
            "sources": sources,
            "needs_clarification": needs_clarification,
            "is_followup": is_followup
        }
        
    except Exception as e:
        logger.error(f"Error generating answer: {str(e)}")
        raise

async def generate_question_suggestions(
    contexts: List[dict], 
    n_suggestions: int, 
    model: str, 
    original_question: str,
    needs_clarification: bool
) -> List[str]:
    try:
        if not contexts:
            return []
            
        formatted_contexts = [
            f"""Content: {chunk["text"]}
Source: {chunk["filename"]} ({chunk["source_type"]})
---""" for chunk in contexts[:5]
        ]
        
        context = "\n".join(formatted_contexts)
        
        prompt_addition = """
6. If multiple client sources were found, suggest more specific questions to help narrow down the information source.""" if needs_clarification else ""

        prompt = f"""Based on ONLY the provided content, generate {n_suggestions} questions.

Content:
{context}

Rules:
1. Questions must be answerable using ONLY the provided content
2. Questions should be different from: "{original_question}"
3. Format as numbered list
4. Questions should be relevant and meaningful
5. Questions should explore different aspects of the content{prompt_addition}"""
        
        response = await openai.ChatCompletion.acreate(
            model=model,
            messages=[
                {"role": "system", "content": "Generate focused follow-up questions based on the content."},
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
        logger.error(f"Question suggestion generation failed: {str(e)}")
        raise

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

        result = await generate_answer(request.question, contexts, request.model)
        
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