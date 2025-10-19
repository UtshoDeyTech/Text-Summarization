from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct, Filter, FieldCondition, MatchValue, PayloadSchemaType
from datetime import datetime
from app.service.log_client import logger
import uuid
from config import (
    QDRANT_URL,
    QDRANT_HOST,
    QDRANT_PORT,
    QDRANT_API_KEY,
    QDRANT_COLLECTION_NAME,
    QDRANT_USE_HTTPS,
    QDRANT_TIMEOUT
)
from typing import List, Dict, Optional

DIMENSION = 1536

# Initialize Qdrant client
def get_qdrant_client():
    """Get Qdrant client instance - supports both URL and host:port configuration"""
    try:
        # Priority 1: Use QDRANT_URL if provided (for production/cloud deployments)
        if QDRANT_URL:
            logger.info(f"Connecting to Qdrant using URL | url={QDRANT_URL}")
            if QDRANT_API_KEY:
                client = QdrantClient(
                    url=QDRANT_URL,
                    api_key=QDRANT_API_KEY,
                    timeout=QDRANT_TIMEOUT
                )
            else:
                client = QdrantClient(
                    url=QDRANT_URL,
                    timeout=QDRANT_TIMEOUT
                )
        # Priority 2: Use host:port (for development/Docker)
        else:
            logger.info(f"Connecting to Qdrant using host:port | host={QDRANT_HOST}, port={QDRANT_PORT}, https={QDRANT_USE_HTTPS}")
            if QDRANT_API_KEY:
                client = QdrantClient(
                    host=QDRANT_HOST,
                    port=QDRANT_PORT,
                    api_key=QDRANT_API_KEY,
                    https=QDRANT_USE_HTTPS,
                    timeout=QDRANT_TIMEOUT
                )
            else:
                client = QdrantClient(
                    host=QDRANT_HOST,
                    port=QDRANT_PORT,
                    https=QDRANT_USE_HTTPS,
                    timeout=QDRANT_TIMEOUT
                )

        # Verify connection
        client.get_collections()
        logger.info(f"Successfully connected to Qdrant")
        return client

    except Exception as e:
        logger.error(f"Failed to connect to Qdrant | error={str(e)}")
        raise

def initialize_qdrant():
    """Initialize Qdrant collection for all documents"""
    try:
        client = get_qdrant_client()

        # Check if collection exists
        collections = client.get_collections().collections
        collection_names = [col.name for col in collections]

        logger.info(f"Initializing Qdrant | collection_name={QDRANT_COLLECTION_NAME}")

        if QDRANT_COLLECTION_NAME not in collection_names:
            logger.info(f"Creating new Qdrant collection | collection_name={QDRANT_COLLECTION_NAME}, dimension={DIMENSION}")
            client.create_collection(
                collection_name=QDRANT_COLLECTION_NAME,
                vectors_config=VectorParams(size=DIMENSION, distance=Distance.COSINE)
            )
            logger.info(f"Created Qdrant collection successfully | collection_name={QDRANT_COLLECTION_NAME}")

            # Create payload indexes for fast filtering (10-100x faster queries)
            logger.info(f"Creating payload indexes for optimized filtering")

            # Index for document_id (used in find_document, delete_vectors, etc.)
            client.create_payload_index(
                collection_name=QDRANT_COLLECTION_NAME,
                field_name="document_id",
                field_schema=PayloadSchemaType.KEYWORD
            )
            logger.info(f"Created index on 'document_id' field")

            # Index for user_id (used in list_documents, list_all_vectors, etc.)
            client.create_payload_index(
                collection_name=QDRANT_COLLECTION_NAME,
                field_name="user_id",
                field_schema=PayloadSchemaType.KEYWORD
            )
            logger.info(f"Created index on 'user_id' field")

            # Index for url (used in URL-based document operations)
            client.create_payload_index(
                collection_name=QDRANT_COLLECTION_NAME,
                field_name="url",
                field_schema=PayloadSchemaType.KEYWORD
            )
            logger.info(f"Created index on 'url' field")

            # Index for content_type (used in list_url_documents)
            client.create_payload_index(
                collection_name=QDRANT_COLLECTION_NAME,
                field_name="content_type",
                field_schema=PayloadSchemaType.KEYWORD
            )
            logger.info(f"Created index on 'content_type' field")

            logger.info(f"All payload indexes created successfully - filtering is now optimized")

        return client
    except Exception as e:
        logger.error(f"Qdrant initialization failed | error_type={type(e).__name__}, error={str(e)}")
        raise

def get_user_documents(user_id: str) -> list:
    """Get all document IDs for a specific user (replaces namespace listing)"""
    try:
        client = get_qdrant_client()

        # Scroll through all points with this user_id
        scroll_result = client.scroll(
            collection_name=QDRANT_COLLECTION_NAME,
            scroll_filter=Filter(
                must=[
                    FieldCondition(
                        key="user_id",
                        match=MatchValue(value=user_id)
                    )
                ]
            ),
            limit=10000,
            with_payload=True,
            with_vectors=False
        )

        # Extract unique document IDs
        document_ids = set()
        for point in scroll_result[0]:
            if point.payload and 'document_id' in point.payload:
                document_ids.add(point.payload['document_id'])

        logger.info(f"Found documents for user | user_id={user_id}, document_count={len(document_ids)}")
        return list(document_ids)

    except Exception as e:
        logger.error(f"Error getting user documents | user_id={user_id}, error={str(e)}")
        return []

def list_documents_from_qdrant(user_id: str) -> list:
    """Get all documents for a user from Qdrant"""
    try:
        logger.info(f"Getting documents from Qdrant | user_id={user_id}")

        client = get_qdrant_client()

        # Scroll through points for this user
        scroll_result = client.scroll(
            collection_name=QDRANT_COLLECTION_NAME,
            scroll_filter=Filter(
                must=[
                    FieldCondition(
                        key="user_id",
                        match=MatchValue(value=user_id)
                    )
                ]
            ),
            limit=10000,
            with_payload=True,
            with_vectors=False
        )

        # Build unique documents list
        documents = {}
        for point in scroll_result[0]:
            if not point.payload:
                continue

            doc_id = point.payload.get('document_id', '')
            if doc_id and doc_id not in documents:
                documents[doc_id] = {
                    "id": doc_id,
                    "name": point.payload.get('filename', doc_id),
                    "type": point.payload.get('file_type', ''),
                    "created_date": point.payload.get('upload_date', ''),
                    "namespace": doc_id  # For compatibility
                }

        document_list = list(documents.values())
        logger.info(f"Retrieved documents from Qdrant | user_id={user_id}, document_count={len(document_list)}")
        return document_list

    except Exception as e:
        logger.error(f"Error listing documents from Qdrant | user_id={user_id}, error={str(e)}")
        return []

def find_document_namespace(document_id: str) -> str:
    """Find if a document exists (returns document_id if found, None otherwise)"""
    try:
        logger.info(f"Finding document | document_id={document_id}")
        client = get_qdrant_client()

        # Check if any points exist with this document_id
        scroll_result = client.scroll(
            collection_name=QDRANT_COLLECTION_NAME,
            scroll_filter=Filter(
                must=[
                    FieldCondition(
                        key="document_id",
                        match=MatchValue(value=document_id)
                    )
                ]
            ),
            limit=1,
            with_payload=False,
            with_vectors=False
        )

        if scroll_result[0]:
            logger.info(f"Found document | document_id={document_id}")
            return document_id

        return None

    except Exception as e:
        logger.error(f"Error finding document | document_id={document_id}, error={str(e)}")
        return None

def upsert_vectors(vectors, metadatas, ids, document_id: str, user_id: str = None):
    """Upsert vectors with metadata into Qdrant"""
    try:
        logger.info(f"Upserting vectors | document_id={document_id}, vector_count={len(vectors)}")

        client = initialize_qdrant()

        # Process in batches of 100 vectors
        batch_size = 100
        points = []

        for i, (vector, metadata, point_id) in enumerate(zip(vectors, metadatas, ids)):
            # Add document_id and user_id to metadata for filtering
            metadata['document_id'] = document_id
            if user_id:
                metadata['user_id'] = user_id
            metadata['upload_date'] = metadata.get('upload_date', datetime.utcnow().isoformat())

            # Create point
            point = PointStruct(
                id=str(uuid.uuid4()),  # Qdrant requires unique IDs
                vector=vector,
                payload=metadata
            )
            points.append(point)

            # Upsert batch when ready
            if len(points) >= batch_size or i == len(vectors) - 1:
                client.upsert(
                    collection_name=QDRANT_COLLECTION_NAME,
                    points=points
                )
                logger.info(f"Batch upsert successful | batch_size={len(points)}, total_progress={i+1}/{len(vectors)}")
                points = []

        logger.info(f"All vectors upserted successfully | document_id={document_id}, total_vectors={len(vectors)}")

    except Exception as e:
        logger.error(f"Vector upsert failed | document_id={document_id}, vector_count={len(vectors)}, error={str(e)}")
        raise

def delete_vectors(document_id: str, vector_ids: list = None):
    """Delete vectors for a document"""
    try:
        logger.info(f"Attempting to delete vectors | document_id={document_id}")

        client = get_qdrant_client()

        # Delete all points with this document_id
        client.delete(
            collection_name=QDRANT_COLLECTION_NAME,
            points_selector=Filter(
                must=[
                    FieldCondition(
                        key="document_id",
                        match=MatchValue(value=document_id)
                    )
                ]
            )
        )

        logger.info(f"Deleted all vectors for document | document_id={document_id}")
        return True

    except Exception as e:
        logger.error(f"Vector deletion failed | document_id={document_id}, error={str(e)}")
        raise

def list_all_vectors(user_id: str):
    """List all vectors for a user"""
    try:
        logger.info(f"Listing all vectors | user_id={user_id}")

        client = get_qdrant_client()

        # Scroll through all points for this user
        scroll_result = client.scroll(
            collection_name=QDRANT_COLLECTION_NAME,
            scroll_filter=Filter(
                must=[
                    FieldCondition(
                        key="user_id",
                        match=MatchValue(value=user_id)
                    )
                ]
            ),
            limit=10000,
            with_payload=True,
            with_vectors=True
        )

        all_matches = scroll_result[0]
        logger.info(f"Vector listing successful | user_id={user_id}, total_vectors={len(all_matches)}")
        return all_matches

    except Exception as e:
        logger.error(f"Vector listing failed | user_id={user_id}, error={str(e)}")
        return []

# URL-specific functions

def find_url_namespace(url: str) -> str:
    """Find if a URL document exists"""
    try:
        logger.info(f"Finding URL document | url={url}")
        client = get_qdrant_client()

        scroll_result = client.scroll(
            collection_name=QDRANT_COLLECTION_NAME,
            scroll_filter=Filter(
                must=[
                    FieldCondition(
                        key="url",
                        match=MatchValue(value=url)
                    )
                ]
            ),
            limit=1,
            with_payload=True,
            with_vectors=False
        )

        if scroll_result[0]:
            # Return document_id as "namespace"
            return scroll_result[0][0].payload.get('document_id', None)

        return None

    except Exception as e:
        logger.error(f"Error finding URL document | url={url}, error={str(e)}")
        return None

def upsert_url_vectors(vectors, metadatas, ids, url: str) -> str:
    """Upsert URL vectors"""
    try:
        # Check if URL already exists and delete if found
        existing_doc_id = find_url_namespace(url)
        if existing_doc_id:
            delete_vectors(existing_doc_id)
            logger.info(f"Deleted existing URL vectors | document_id={existing_doc_id}")

        # Create new document_id for URL
        document_id = str(uuid.uuid4())
        logger.info(f"Upserting URL vectors | document_id={document_id}, vector_count={len(vectors)}")

        client = initialize_qdrant()
        points = []

        for vector, metadata, point_id in zip(vectors, metadatas, ids):
            metadata['document_id'] = document_id
            metadata['url'] = url

            point = PointStruct(
                id=str(uuid.uuid4()),
                vector=vector,
                payload=metadata
            )
            points.append(point)

        # Upsert all points
        client.upsert(
            collection_name=QDRANT_COLLECTION_NAME,
            points=points
        )

        logger.info(f"URL vector upsert successful | document_id={document_id}, vector_count={len(vectors)}")
        return document_id

    except Exception as e:
        logger.error(f"URL vector upsert failed | url={url}, error={str(e)}")
        raise

def list_url_documents() -> list:
    """List all URL documents"""
    try:
        logger.info("Listing URL documents")

        client = get_qdrant_client()

        scroll_result = client.scroll(
            collection_name=QDRANT_COLLECTION_NAME,
            scroll_filter=Filter(
                must=[
                    FieldCondition(
                        key="content_type",
                        match=MatchValue(value="url")
                    )
                ]
            ),
            limit=10000,
            with_payload=True,
            with_vectors=False
        )

        documents = {}
        for point in scroll_result[0]:
            if not point.payload:
                continue

            url = point.payload.get('url')
            if url and url not in documents:
                documents[url] = {
                    "url": url,
                    "upload_date": point.payload.get('upload_date'),
                    "namespace": point.payload.get('document_id')
                }

        document_list = list(documents.values())
        logger.info(f"Retrieved URL documents | document_count={len(document_list)}")
        return document_list

    except Exception as e:
        logger.error(f"Error listing URL documents | error={str(e)}")
        return []
