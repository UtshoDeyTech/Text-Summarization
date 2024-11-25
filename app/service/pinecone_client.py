import os 
from pinecone import Pinecone, ServerlessSpec
from datetime import datetime
from app.get_secret_key import get_secret
from app.service.log_client import logger
import uuid
from dotenv import load_dotenv

load_dotenv()


PINECONE_API_KEY = get_secret("PINECONE_API_KEY")
PINECONE_ENVIRONMENT = get_secret("PINECONE_ENVIRONMENT")
DIMENSION = 1536
PINECONE_CLIENT_INDEX = "client-document"

pc = Pinecone(api_key=PINECONE_API_KEY)

def initialize_pinecone():
    """Initialize a single Pinecone index for all users"""
    try:
        existing_indexes = pc.list_indexes().names()
        
        logger.info(f"Initializing Pinecone | index_name={PINECONE_CLIENT_INDEX}")
        if PINECONE_CLIENT_INDEX not in existing_indexes:
            logger.info(f"Creating new Pinecone index | index_name={PINECONE_CLIENT_INDEX}, dimension={DIMENSION}")
            pc.create_index(
                name=PINECONE_CLIENT_INDEX,
                dimension=DIMENSION,
                metric="cosine",
                spec=ServerlessSpec(
                    cloud="aws",
                    region="us-west-2"
                )
            )
        return pc.Index(PINECONE_CLIENT_INDEX)
    except Exception as e:
        logger.error(f"Pinecone initialization failed | error_type={type(e).__name__}, error={str(e)}")
        raise

def get_user_namespaces(user_id: str) -> list:
    """Get all namespaces for a specific user."""
    try:
        index = initialize_pinecone()
        stats = index.describe_index_stats()
        user_namespaces = [ns for ns in stats.namespaces.keys() if ns.startswith(f"{user_id}_")]
        logger.info(f"Found namespaces | user_id={user_id}, namespace_count={len(user_namespaces)}")
        return user_namespaces
    except Exception as e:
        logger.error(f"Error getting user namespaces | user_id={user_id}, error={str(e)}")
        return []

def get_namespace(user_id: str, document_id: str = None):
    """Generate namespace for a user's documents"""
    if document_id:
        return f"{user_id}_{uuid.uuid4()}"
    return f"{user_id}"

def upsert_vectors(user_id: str, vectors, metadatas, ids, document_id: str):
    """Upsert vectors with metadata into Pinecone."""
    try:
        namespace = get_namespace(user_id, document_id)
        logger.info(f"Upserting vectors | namespace={namespace}, vector_count={len(vectors)}")
        
        for metadata in metadatas:
            metadata['upload_date'] = metadata.get('upload_date', datetime.utcnow().isoformat())
            metadata['namespace'] = namespace
                
        index = initialize_pinecone()
        index.upsert(
            vectors=list(zip(ids, vectors, metadatas)),
            namespace=namespace
        )
        logger.info(f"Vector upsert successful | namespace={namespace}, vector_count={len(vectors)}")
    except Exception as e:
        logger.error(f"Vector upsert failed | user_id={user_id}, vector_count={len(vectors)}, error={str(e)}")
        raise

def find_document_namespace(user_id: str, document_id: str) -> str:
    """Find the namespace containing a specific document."""
    try:
        logger.info(f"Finding namespace for document | user_id={user_id}, document_id={document_id}")
        
        user_namespaces = get_user_namespaces(user_id)
        if not user_namespaces:
            return None
            
        index = initialize_pinecone()
        
        for namespace in user_namespaces:
            try:
                results = index.query(
                    vector=[0] * DIMENSION,
                    top_k=1,
                    namespace=namespace,
                    include_metadata=True
                )
                
                if not results.matches:
                    continue
                    
                metadata = results.matches[0].metadata
                if metadata:
                    stored_doc_id = metadata.get('document_id', '')
                    stored_filename = metadata.get('filename', '')
                    
                    if stored_doc_id == document_id or stored_filename == document_id:
                        logger.info(f"Found document in namespace | namespace={namespace}")
                        return namespace
                        
            except Exception as e:
                logger.error(f"Error checking namespace | namespace={namespace}, error={str(e)}")
                continue
                
        return None
        
    except Exception as e:
        logger.error(f"Error finding document namespace | error={str(e)}")
        return None

def list_documents_from_pinecone(user_id: str) -> list:
    """Get all documents for a user from Pinecone metadata."""
    try:
        logger.info(f"Getting documents from Pinecone | user_id={user_id}")
        
        user_namespaces = get_user_namespaces(user_id)
        if not user_namespaces:
            return []

        index = initialize_pinecone()
        documents = {}
        
        for namespace in user_namespaces:
            try:
                results = index.query(
                    vector=[0] * DIMENSION,
                    top_k=1,
                    namespace=namespace,
                    include_metadata=True
                )
                
                if not results.matches:
                    continue
                    
                metadata = results.matches[0].metadata
                if metadata:
                    doc_id = metadata.get('document_id', '')
                    if doc_id and doc_id not in documents:
                        documents[doc_id] = {
                            "id": doc_id,
                            "name": metadata.get('filename', doc_id),
                            "type": metadata.get('file_type', ''),
                            "created_date": metadata.get('upload_date', ''),
                            "namespace": namespace
                        }
                        
            except Exception as e:
                logger.error(f"Error checking namespace | namespace={namespace}, error={str(e)}")
                continue
                
        document_list = list(documents.values())
        logger.info(f"Retrieved documents from Pinecone | user_id={user_id}, document_count={len(document_list)}")
        return document_list
        
    except Exception as e:
        logger.error(f"Error listing documents from Pinecone | user_id={user_id}, error={str(e)}")
        return []
    

def list_all_vectors(user_id: str):
    """List all vectors for a user across all their namespaces."""
    try:
        logger.info(f"Listing all vectors | user_id={user_id}")
        
        # Get all namespaces for this user
        user_namespaces = get_user_namespaces(user_id)
        if not user_namespaces:
            logger.warning(f"No namespaces found | user_id={user_id}")
            return []
            
        # Initialize Pinecone client
        index = initialize_pinecone()
        
        # Query each namespace and collect all vectors
        all_matches = []
        for curr_namespace in user_namespaces:
            try:
                results = index.query(
                    vector=[0] * DIMENSION,
                    top_k=10000,
                    namespace=curr_namespace,
                    include_metadata=True
                )
                all_matches.extend(results.matches)
            except Exception as e:
                logger.error(f"Error querying namespace | namespace={curr_namespace}, error={str(e)}")
                continue
        
        logger.info(f"Vector listing successful | user_id={user_id}, total_vectors={len(all_matches)}")
        return all_matches
        
    except Exception as e:
        logger.error(f"Vector listing failed | user_id={user_id}, error={str(e)}")
        return []

def delete_vectors(user_id: str, document_id: str, vector_ids: list = None):
    """Delete vectors for a document, either by namespace or specific IDs."""
    try:
        logger.info(f"Attempting to delete vectors | user_id={user_id}, document_id={document_id}")
        
        # Find the namespace containing the document
        namespace = find_document_namespace(user_id, document_id)
        if not namespace:
            logger.warning(f"No namespace found for document | document_id={document_id}")
            return False
            
        index = initialize_pinecone()
        
        # If specific vector IDs are provided, delete only those
        if vector_ids:
            try:
                index.delete(ids=vector_ids, namespace=namespace)
                logger.info(f"Deleted specific vectors | namespace={namespace}, vector_count={len(vector_ids)}")
                return True
            except Exception as e:
                logger.error(f"Error deleting vectors | namespace={namespace}, error={str(e)}")
                raise
        # Otherwise delete the entire namespace
        else:
            try:
                index.delete(delete_all=True, namespace=namespace)
                logger.info(f"Deleted entire namespace | namespace={namespace}")
                return True
            except Exception as e:
                logger.error(f"Error deleting namespace | namespace={namespace}, error={str(e)}")
                raise
                
    except Exception as e:
        logger.error(f"Vector deletion failed | user_id={user_id}, document_id={document_id}, error={str(e)}")
        raise