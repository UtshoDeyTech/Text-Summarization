import logging
from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import JSONResponse
from app.service.s3_storage import list_objects, S3_BUCKET_NAME, s3_client
from app.service.log_client import logger
from app.service.qdrant_client import list_documents_from_qdrant
from datetime import datetime

router = APIRouter()

SUPPORTED_EXTENSIONS = {
    'pdf': 'application/pdf',
    'docx': 'application/vnd.openxmlformats-officedocument.wordprocessingml.document'
}

def get_object_creation_date(bucket: str, key: str) -> str:
    try:
        response = s3_client.head_object(Bucket=bucket, Key=key)
        return response['LastModified'].isoformat()
    except Exception as e:
        logger.error(f"Failed to get creation date | bucket={bucket}, key={key}, error={str(e)}")
        return "No Date Found"

@router.get("/{user_id}/list_documents")
async def list_documents(request: Request, user_id: str):
    try:
        logger.info(f"Starting document listing | user_id={user_id}")
        
        # Get documents from Qdrant
        qdrant_docs = list_documents_from_qdrant(user_id)
        logger.info(f"Found {len(qdrant_docs)} documents in Qdrant")
        
        # Get documents from S3
        s3_docs = []
        response = list_objects(S3_BUCKET_NAME)
        
        for obj in response.get('Contents', []):
            file_extension = obj['Key'].split('.')[-1].lower() if '.' in obj['Key'] else None
            if file_extension in SUPPORTED_EXTENSIONS:
                path_parts = obj['Key'].split('/')
                if len(path_parts) > 1 and path_parts[0] == user_id:
                    creation_date = get_object_creation_date(S3_BUCKET_NAME, obj['Key'])
                    s3_docs.append({
                        "id": path_parts[1],
                        "name": path_parts[1],
                        "type": file_extension,
                        "created_date": creation_date,
                        "storage": "s3"
                    })
        
        logger.info(f"Found {len(s3_docs)} documents in S3")
        
        # Combine and deduplicate documents
        all_docs = {}

        # Add Qdrant documents
        for doc in qdrant_docs:
            doc_id = doc['id']
            all_docs[doc_id] = {**doc, "storage_locations": ["qdrant"]}
        
        # Add or update with S3 documents
        for doc in s3_docs:
            doc_id = doc['id']
            if doc_id in all_docs:
                all_docs[doc_id]["storage_locations"].append("s3")
                # Update creation date if S3 date is earlier
                if doc["created_date"] < all_docs[doc_id]["created_date"]:
                    all_docs[doc_id]["created_date"] = doc["created_date"]
            else:
                all_docs[doc_id] = {**doc, "storage_locations": ["s3"]}
        
        # Convert to list
        documents = list(all_docs.values())
        
        logger.info(f"Document listing completed | user_id={user_id}, total_documents={len(documents)}")
        
        return JSONResponse(content={
            "user_id": user_id,
            "total_documents": len(documents),
            "documents": documents,
            "status_code": "200"
        })
        
    except Exception as e:
        logger.error(f"Document listing failed | user_id={user_id}, error={str(e)}")
        raise HTTPException(
            status_code=500,
            detail={"status_code": "500", "error_messages": [str(e)]}
        )