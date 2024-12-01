from pinecone import Pinecone, ServerlessSpec
from datetime import datetime
import uuid
from app.service.log_client import logger
from config import PINECONE_API_KEY, PINECONE_ANC_INDEX

DIMENSION = 1536
pc = Pinecone(api_key=PINECONE_API_KEY)

def initialize_url_pinecone():
    """Initialize Pinecone index for URL content"""
    try:
        existing_indexes = pc.list_indexes().names()
        
        if PINECONE_ANC_INDEX not in existing_indexes:
            logger.info(f"Creating new Pinecone index | index_name={PINECONE_ANC_INDEX}")
            pc.create_index(
                name=PINECONE_ANC_INDEX,
                dimension=DIMENSION,
                metric="cosine",
                spec=ServerlessSpec(
                    cloud="aws",
                    region="us-west-2"
                )
            )
            logger.info(f"Created new Pinecone index | index_name={PINECONE_ANC_INDEX}")
            
        logger.info(f"Initializing Pinecone for URLs | index_name={PINECONE_ANC_INDEX}")
        return pc.Index(PINECONE_ANC_INDEX)
    except Exception as e:
        logger.error(f"Pinecone URL initialization failed | error={str(e)}")
        raise

def get_url_namespaces() -> list:
    """Get all URL namespaces"""
    try:
        index = initialize_url_pinecone()
        stats = index.describe_index_stats()
        url_namespaces = list(stats.namespaces.keys())
        logger.info(f"Found URL namespaces | namespace_count={len(url_namespaces)}")
        return url_namespaces
    except Exception as e:
        logger.error(f"Error getting URL namespaces | error={str(e)}")
        return []

def find_url_namespace(url: str) -> str:
    """Find namespace containing specific URL content"""
    try:
        logger.info(f"Finding namespace for URL | url={url}")
        
        namespaces = get_url_namespaces()
        if not namespaces:
            return None
            
        index = initialize_url_pinecone()
        
        for namespace in namespaces:
            try:
                results = index.query(
                    vector=[0] * DIMENSION,
                    top_k=1,
                    namespace=namespace,
                    include_metadata=True
                )
                
                if results.matches:
                    metadata = results.matches[0].metadata
                    if metadata.get('url') == url:
                        logger.info(f"Found URL in namespace | namespace={namespace}")
                        return namespace
                        
            except Exception as e:
                logger.error(f"Error checking namespace | namespace={namespace}, error={str(e)}")
                continue
                
        return None
        
    except Exception as e:
        logger.error(f"Error finding URL namespace | error={str(e)}")
        return None

def upsert_url_vectors(vectors, metadatas, ids, url: str):
    """Upsert vectors with metadata into Pinecone for URL content"""
    try:
        # Check if URL already exists and delete if found
        existing_namespace = find_url_namespace(url)
        if existing_namespace:
            index = initialize_url_pinecone()
            index.delete(delete_all=True, namespace=existing_namespace)
            logger.info(f"Deleted existing URL vectors | namespace={existing_namespace}")
        
        # Create new namespace for URL
        namespace = str(uuid.uuid4())
        logger.info(f"Upserting URL vectors | namespace={namespace}, vector_count={len(vectors)}")
        
        # Add namespace to metadata
        for metadata in metadatas:
            metadata['namespace'] = namespace
        
        index = initialize_url_pinecone()
        index.upsert(
            vectors=list(zip(ids, vectors, metadatas)),
            namespace=namespace
        )
        
        logger.info(f"URL vector upsert successful | namespace={namespace}, vector_count={len(vectors)}")
        
    except Exception as e:
        logger.error(f"URL vector upsert failed | url={url}, error={str(e)}")
        raise

def list_url_documents() -> list:
    """List all URL documents"""
    try:
        logger.info("Listing URL documents")
        
        namespaces = get_url_namespaces()
        if not namespaces:
            return []
            
        index = initialize_url_pinecone()
        documents = {}
        
        for namespace in namespaces:
            try:
                results = index.query(
                    vector=[0] * DIMENSION,
                    top_k=1,
                    namespace=namespace,
                    include_metadata=True
                )
                
                if results.matches:
                    metadata = results.matches[0].metadata
                    url = metadata.get('url')
                    if url and url not in documents:
                        documents[url] = {
                            "url": url,
                            "upload_date": metadata.get('upload_date'),
                            "namespace": namespace
                        }
                        
            except Exception as e:
                logger.error(f"Error checking namespace | namespace={namespace}, error={str(e)}")
                continue
                
        document_list = list(documents.values())
        logger.info(f"Retrieved URL documents | document_count={len(document_list)}")
        return document_list
        
    except Exception as e:
        logger.error(f"Error listing URL documents | error={str(e)}")
        return []