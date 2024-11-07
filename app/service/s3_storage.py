import os
import logging
import boto3
from botocore.exceptions import ClientError
from app.get_secret_key import get_secret
from app.service.log_client import logger

# Get S3 configuration
S3_REGION_NAME = get_secret("S3_REGION_NAME")
S3_END_POINT_URL = get_secret("S3_END_POINT_URL")
S3_ACCESS_KEY = get_secret("S3_ACCESS_KEY")
S3_SECRET_KEY = get_secret("S3_SECRET_KEY")
S3_BUCKET_NAME = get_secret("S3_BUCKET_NAME")

logger.info(f"Initializing S3 client | endpoint={S3_END_POINT_URL}, region={S3_REGION_NAME}")

s3_client = boto3.client(
    "s3",
    region_name=S3_REGION_NAME,
    endpoint_url=S3_END_POINT_URL,
    aws_access_key_id=S3_ACCESS_KEY,
    aws_secret_access_key=S3_SECRET_KEY,
)

def verify_bucket(bucket):
    """Verify bucket exists and is accessible"""
    try:
        s3_client.head_bucket(Bucket=bucket)
        logger.info(f"Bucket verification successful | bucket={bucket}")
        return True
    except ClientError as e:
        error_code = e.response.get('Error', {}).get('Code', 'Unknown')
        logger.error(f"Bucket verification failed | bucket={bucket}, error_code={error_code}, error={str(e)}")
        return False

def upload_file(file_obj, bucket, object_name, expiration=3600):
    """Upload a file to S3 bucket and return presigned URL"""
    try:
        logger.info(f"Starting file upload | bucket={bucket}, object={object_name}, size={file_obj.size if hasattr(file_obj, 'size') else 'unknown'}")
        s3_client.upload_fileobj(file_obj, bucket, object_name)
        
        logger.info(f"Generating presigned URL | bucket={bucket}, object={object_name}, expiration={expiration}")
        response = s3_client.generate_presigned_url(
            "get_object",
            Params={"Bucket": bucket, "Key": object_name},
            ExpiresIn=expiration,
        )
        
        logger.info(f"File upload successful | bucket={bucket}, object={object_name}")
        return response
        
    except ClientError as e:
        error_code = e.response.get('Error', {}).get('Code', 'Unknown')
        logger.error(f"File upload failed | bucket={bucket}, object={object_name}, error_code={error_code}, error={str(e)}")
        return None

def list_objects(bucket, prefix=None):
    """List objects in S3 bucket with optional prefix"""
    try:
        params = {'Bucket': bucket}
        if prefix:
            params['Prefix'] = prefix
            
        logger.info(f"Listing objects | bucket={bucket}, prefix={prefix or 'none'}")
        
        response = s3_client.list_objects_v2(**params)
        object_count = len(response.get('Contents', []))
        logger.info(f"Object listing successful | bucket={bucket}, count={object_count}")
        return response
        
    except ClientError as e:
        error_code = e.response.get('Error', {}).get('Code', 'Unknown')
        logger.error(f"Object listing failed | bucket={bucket}, prefix={prefix or 'none'}, error_code={error_code}, error={str(e)}")
        raise

def delete_object(bucket, object_name):
    """Delete an object from S3 bucket"""
    try:
        logger.info(f"Starting object deletion | bucket={bucket}, object={object_name}")
        s3_client.delete_object(Bucket=bucket, Key=object_name)
        logger.info(f"Object deletion successful | bucket={bucket}, object={object_name}")
        
    except ClientError as e:
        error_code = e.response.get('Error', {}).get('Code', 'Unknown')
        logger.error(f"Object deletion failed | bucket={bucket}, object={object_name}, error_code={error_code}, error={str(e)}")
        raise

def get_object_size(bucket, object_name):
    """Get size of an object in S3 bucket"""
    try:
        logger.info(f"Getting object size | bucket={bucket}, object={object_name}")
        response = s3_client.head_object(Bucket=bucket, Key=object_name)
        size = response.get('ContentLength', 0)
        logger.info(f"Object size retrieved | bucket={bucket}, object={object_name}, size={size}")
        return size
    except ClientError as e:
        error_code = e.response.get('Error', {}).get('Code', 'Unknown')
        logger.error(f"Failed to get object size | bucket={bucket}, object={object_name}, error_code={error_code}, error={str(e)}")
        raise