import os
import logging
from pinecone import Pinecone, ServerlessSpec
from app.get_secret_key import get_secret

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load environment variables
PINECONE_API_KEY = get_secret("PINECONE_API_KEY")
PINECONE_ENVIRONMENT = get_secret("PINECONE_ENVIRONMENT")

INDEX_NAME = "pdf-vectors"
DIMENSION = 1536  # Assuming you're using OpenAI's text-embedding-ada-002 model

pc = Pinecone(api_key=PINECONE_API_KEY)

def initialize_pinecone():
    try:
        if INDEX_NAME not in pc.list_indexes().names():
            pc.create_index(
                name=INDEX_NAME,
                dimension=DIMENSION,
                metric="cosine",
                spec=ServerlessSpec(
                    cloud="aws",
                    region="us-west-2"
                )
            )
        return pc.Index(INDEX_NAME)
    except Exception as e:
        logger.error(f"Error initializing Pinecone: {str(e)}")
        raise

def upsert_vectors(index, vectors, metadatas, ids):
    try:
        index.upsert(vectors=list(zip(ids, vectors, metadatas)))
    except Exception as e:
        logger.error(f"Error upserting vectors: {str(e)}")
        raise

def query_vectors(index, query_vector, top_k=5):
    try:
        results = index.query(vector=query_vector, top_k=top_k, include_metadata=True)
        return results.matches
    except Exception as e:
        logger.error(f"Error querying vectors: {str(e)}")
        raise

def delete_vectors(index, ids):
    try:
        if not ids:
            logger.warning("No ids provided for deletion")
            return
        
        logger.info(f"Attempting to delete {len(ids)} vectors")
        index.delete(ids=ids)
        logger.info(f"Successfully deleted {len(ids)} vectors")
    except Exception as e:
        logger.error(f"Error deleting vectors: {str(e)}")
        raise

def list_all_vectors(index):
    try:
        # Note: This method might not work for large datasets
        results = index.query(vector=[0] * DIMENSION, top_k=10000, include_metadata=True)
        logger.info(f"Retrieved {len(results.matches)} vectors from Pinecone")
        return results.matches
    except Exception as e:
        logger.error(f"Error listing vectors: {str(e)}")
        raise

# Test function
def test_pinecone_connection():
    try:
        index = initialize_pinecone()
        print(f"Successfully connected to Pinecone index: {INDEX_NAME}")
        stats = index.describe_index_stats()
        print(f"Index stats: {stats}")
    except Exception as e:
        print(f"Error connecting to Pinecone: {str(e)}")

if __name__ == "__main__":
    test_pinecone_connection()