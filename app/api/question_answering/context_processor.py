from typing import List, Union
from pinecone import Pinecone
from datetime import datetime, timedelta
from app.service.openai_client import get_embeddings
from app.service.log_client import logger
from config import PINECONE_CLIENT_INDEX, PINECONE_ANC_INDEX, PINECONE_API_KEY, VALIDATION_DAYS

def validate_days(days: Union[str, int]) -> int:
    try:
        return int(days)
    except (ValueError, TypeError):
        logger.warning(f"Invalid VALIDATION_DAYS value: {days}. Using default of 365 days.")
        return 365

async def get_context_from_vectors(question: str, user_id: str, max_chunks: int = 10) -> List[dict]:
    try:
        question_embedding = get_embeddings([question])[0]
        current_date = datetime.utcnow()
        validation_days = validate_days(VALIDATION_DAYS)
        twelve_months_ago = current_date - timedelta(days=validation_days)
        
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
                if "upload_date" not in metadata:
                    continue
                    
                upload_date = datetime.fromisoformat(metadata["upload_date"])
                if upload_date >= twelve_months_ago:
                    context = {
                        "text": metadata.get("text", ""),
                        "filename": metadata.get("filename", ""),
                        "document_id": metadata.get("document_id", ""),
                        "file_type": metadata.get("file_type", ""),
                        "source_type": "client",
                        "namespace": metadata.get("namespace", ""),
                        "org_id": metadata.get("org_id", ""),
                        "docs_category": metadata.get("docs_category", ""),
                        "user_id": metadata.get("user_id", ""),
                        "score": match.score,
                        "upload_date": upload_date.isoformat()
                    }
                    if context["text"].strip():
                        client_contexts.append(context)

        global_index = pc.Index(PINECONE_ANC_INDEX)
        global_results = global_index.query(
            vector=question_embedding,
            top_k=max_chunks,
            include_metadata=True
        )
        
        global_contexts = []
        for match in global_results.matches:
            metadata = match.metadata or {}
            if "upload_date" not in metadata:
                continue
                
            upload_date = datetime.fromisoformat(metadata["upload_date"])
            if upload_date >= twelve_months_ago:
                context = {
                    "text": metadata.get("text", ""),
                    "filename": metadata.get("url", ""),
                    "document_id": metadata.get("document_id", ""),
                    "file_type": "url",
                    "source_type": "global",
                    "score": match.score,
                    "upload_date": upload_date.isoformat()
                }
                if context["text"].strip():
                    global_contexts.append(context)

        all_contexts = sorted(client_contexts + global_contexts, 
                            key=lambda x: (x["source_type"] != "client", -x["score"]))
                            
        return all_contexts[:max_chunks]
        
    except Exception as e:
        logger.error(f"Error getting contexts | user_id={user_id}, error={str(e)}")
        raise