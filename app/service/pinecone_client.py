import os 
from pinecone import Pinecone, ServerlessSpec
from app.get_secret_key import get_secret
from app.service.log_client import logger

PINECONE_API_KEY = get_secret("PINECONE_API_KEY")
PINECONE_ENVIRONMENT = get_secret("PINECONE_ENVIRONMENT")
DIMENSION = 1536

pc = Pinecone(api_key=PINECONE_API_KEY)

def get_index_name(user_id: str) -> str:
    return f"document-vectors-{user_id}"  # Changed from pdf-vectors

def initialize_pinecone(user_id: str):
    try:
        index_name = get_index_name(user_id)
        existing_indexes = pc.list_indexes().names()
        
        logger.info(f"Initializing Pinecone | user_id={user_id}, index_name={index_name}, exists={index_name in existing_indexes}")
        
        if index_name not in existing_indexes:
            logger.info(f"Creating new Pinecone index | index_name={index_name}, dimension={DIMENSION}")
            pc.create_index(
                name=index_name,
                dimension=DIMENSION,
                metric="cosine",
                spec=ServerlessSpec(
                    cloud="aws",
                    region="us-west-2"
                )
            )
        return pc.Index(index_name)
    except Exception as e:
        logger.error(f"Pinecone initialization failed | user_id={user_id}, error_type={type(e).__name__}, error={str(e)}")
        raise

def upsert_vectors(user_id: str, vectors, metadatas, ids):
    try:
        logger.info(f"Upserting vectors | user_id={user_id}, vector_count={len(vectors)}")
        index = initialize_pinecone(user_id)
        index.upsert(vectors=list(zip(ids, vectors, metadatas)))
        logger.info(f"Vector upsert successful | user_id={user_id}, vector_count={len(vectors)}")
    except Exception as e:
        logger.error(f"Vector upsert failed | user_id={user_id}, vector_count={len(vectors)}, error_type={type(e).__name__}, error={str(e)}")
        raise

def query_vectors(user_id: str, query_vector, top_k=5):
    try:
        logger.info(f"Querying vectors | user_id={user_id}, top_k={top_k}")
        index = initialize_pinecone(user_id)
        results = index.query(vector=query_vector, top_k=top_k, include_metadata=True)
        logger.info(f"Vector query successful | user_id={user_id}, matches_found={len(results.matches)}")
        return results.matches
    except Exception as e:
        logger.error(f"Vector query failed | user_id={user_id}, top_k={top_k}, error_type={type(e).__name__}, error={str(e)}")
        raise

def delete_vectors(user_id: str, ids):
    try:
        if not ids:
            logger.info(f"No vectors to delete | user_id={user_id}")
            return
        
        logger.info(f"Deleting vectors | user_id={user_id}, vector_count={len(ids)}")
        index = initialize_pinecone(user_id)
        index.delete(ids=ids)
        logger.info(f"Vector deletion successful | user_id={user_id}, vector_count={len(ids)}")
    except Exception as e:
        logger.error(f"Vector deletion failed | user_id={user_id}, vector_count={len(ids) if ids else 0}, error_type={type(e).__name__}, error={str(e)}")
        raise

def list_all_vectors(user_id: str):
    try:
        logger.info(f"Listing all vectors | user_id={user_id}")
        index = initialize_pinecone(user_id)
        results = index.query(vector=[0] * DIMENSION, top_k=10000, include_metadata=True)
        logger.info(f"Vector listing successful | user_id={user_id}, vector_count={len(results.matches)}")
        return results.matches
    except Exception as e:
        logger.error(f"Vector listing failed | user_id={user_id}, error_type={type(e).__name__}, error={str(e)}")
        raise