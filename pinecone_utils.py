import os
import logging
from dotenv import load_dotenv
import pinecone
from pinecone import ServerlessSpec

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")

pc = pinecone.Pinecone(api_key=PINECONE_API_KEY)

INDEX_NAME = "pdf-vectors"
DIMENSION = 1536  # Assuming you're using OpenAI's text-embedding-ada-002 model

def initialize_pinecone():
    if INDEX_NAME not in pc.list_indexes().names():
        pc.create_index(
            name=INDEX_NAME,
            dimension=DIMENSION,
            metric="cosine",
            spec=ServerlessSpec(
                cloud="aws",
                region="us-east-1"
            )
        )
    return pc.Index(INDEX_NAME)

def upsert_vectors(index, vectors, metadatas, ids):
    index.upsert(vectors=zip(ids, vectors, metadatas))

def query_vectors(index, query_vector, top_k=5):
    results = index.query(vector=query_vector, top_k=top_k, include_metadata=True)
    return results.matches

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
        results = index.query(vector=[0] * DIMENSION, top_k=10000, include_metadata=True)
        logger.info(f"Retrieved {len(results.matches)} vectors from Pinecone")
        return results.matches
    except Exception as e:
        logger.error(f"Error listing vectors: {str(e)}")
        raise