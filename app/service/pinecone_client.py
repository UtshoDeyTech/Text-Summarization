import os
from pinecone import Pinecone, ServerlessSpec
from app.get_secret_key import get_secret
from app.service.log_client import logger

PINECONE_API_KEY = get_secret("PINECONE_API_KEY")
PINECONE_ENVIRONMENT = get_secret("PINECONE_ENVIRONMENT")
DIMENSION = 1536

pc = Pinecone(api_key=PINECONE_API_KEY)

def get_index_name(user_id: str) -> str:
    return f"pdf-vectors-{user_id}"

def initialize_pinecone(user_id: str):
    try:
        index_name = get_index_name(user_id)
        if index_name not in pc.list_indexes().names():
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
        logger.error(f"Error initializing Pinecone for user {user_id}: {str(e)}")
        raise

def upsert_vectors(user_id: str, vectors, metadatas, ids):
    try:
        index = initialize_pinecone(user_id)
        index.upsert(vectors=list(zip(ids, vectors, metadatas)))
    except Exception as e:
        logger.error(f"Error upserting vectors for user {user_id}: {str(e)}")
        raise

def query_vectors(user_id: str, query_vector, top_k=5):
    try:
        index = initialize_pinecone(user_id)
        results = index.query(vector=query_vector, top_k=top_k, include_metadata=True)
        return results.matches
    except Exception as e:
        logger.error(f"Error querying vectors for user {user_id}: {str(e)}")
        raise

def delete_vectors(user_id: str, ids):
    try:
        if not ids:
            return
        index = initialize_pinecone(user_id)
        index.delete(ids=ids)
    except Exception as e:
        logger.error(f"Error deleting vectors for user {user_id}: {str(e)}")
        raise

def list_all_vectors(user_id: str):
    try:
        index = initialize_pinecone(user_id)
        results = index.query(vector=[0] * DIMENSION, top_k=10000, include_metadata=True)
        return results.matches
    except Exception as e:
        logger.error(f"Error listing vectors for user {user_id}: {str(e)}")
        raise