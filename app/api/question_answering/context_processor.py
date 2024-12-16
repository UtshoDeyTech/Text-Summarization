from typing import List
from pinecone import Pinecone
from app.service.openai_client import get_embeddings
from app.service.log_client import logger
from config import PINECONE_CLIENT_INDEX, PINECONE_ANC_INDEX, PINECONE_API_KEY

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