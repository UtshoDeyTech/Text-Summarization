import logging
from fastapi import APIRouter, HTTPException, Header, Path, Request
from fastapi.responses import JSONResponse
from app.service.s3_storage import list_objects, S3_BUCKET_NAME, s3_client
from app.service.log_client import logger
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
async def list_documents(request: Request, 
                         user_id: str):
    try:
        logger.info(f"Starting document listing | user_id={user_id}")
        
        response = list_objects(S3_BUCKET_NAME)
        documents = []
        
        for obj in response.get('Contents', []):
            file_extension = obj['Key'].split('.')[-1].lower() if '.' in obj['Key'] else None
            if file_extension in SUPPORTED_EXTENSIONS:
                path_parts = obj['Key'].split('/')
                if len(path_parts) > 1 and path_parts[0] == user_id:
                    creation_date = get_object_creation_date(S3_BUCKET_NAME, obj['Key'])
                    documents.append({
                        "id": path_parts[1],
                        "name": path_parts[1],
                        "type": file_extension,
                        "created_date": creation_date
                    })
        
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