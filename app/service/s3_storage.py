import os
import logging
import boto3
from botocore.exceptions import ClientError
from app.get_secret_key import get_secret
from app.service.log_client import logger

# Get S3 configuration
S3_REGION_NAME = get_secret("S3_REGION_NAME")
S3_END_POINT_URL = "https://s3.us-east-1.amazonaws.com"  # Standard endpoint
S3_ACCESS_KEY = get_secret("S3_ACCESS_KEY")
S3_SECRET_KEY = get_secret("S3_SECRET_KEY")
S3_BUCKET_NAME = "python-api-ai-dev"  # Your actual bucket name

logger.info(f"Initializing S3 client with endpoint: {S3_END_POINT_URL}, region: {S3_REGION_NAME}")

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
        logger.info(f"Successfully verified bucket: {bucket}")
        return True
    except ClientError as e:
        error_code = e.response.get('Error', {}).get('Code', 'Unknown')
        logger.error(f"Bucket verification failed: {error_code}", 
                    error=str(e), 
                    bucket=bucket)
        return False

def upload_file(file_obj, bucket, object_name, expiration=3600):
    """Upload a file to S3 bucket and return presigned URL"""
    try:
        logger.info(f"Uploading file to S3: {object_name}", bucket=bucket)
        s3_client.upload_fileobj(file_obj, bucket, object_name)
        
        logger.info(f"Generating presigned URL for: {object_name}")
        response = s3_client.generate_presigned_url(
            "get_object",
            Params={"Bucket": bucket, "Key": object_name},
            ExpiresIn=expiration,
        )
        return response
        
    except ClientError as e:
        error_code = e.response.get('Error', {}).get('Code', 'Unknown')
        logger.error(f"S3 upload error: {error_code}", 
                    error=str(e), 
                    bucket=bucket, 
                    object_name=object_name)
        return None

def list_objects(bucket, prefix=None):
    """List objects in S3 bucket with optional prefix"""
    try:
        params = {'Bucket': bucket}
        if prefix:
            params['Prefix'] = prefix
            
        logger.info(f"Listing objects in bucket", bucket=bucket, prefix=prefix or "none")
        
        response = s3_client.list_objects_v2(**params)
        return response
        
    except ClientError as e:
        error_code = e.response.get('Error', {}).get('Code', 'Unknown')
        logger.error(f"S3 list error: {error_code}", 
                    error=str(e), 
                    bucket=bucket, 
                    prefix=prefix)
        raise

def delete_object(bucket, object_name):
    """Delete an object from S3 bucket"""
    try:
        logger.info(f"Deleting object from S3: {object_name}", bucket=bucket)
        s3_client.delete_object(Bucket=bucket, Key=object_name)
        logger.info(f"Successfully deleted object: {object_name}", bucket=bucket)
        
    except ClientError as e:
        error_code = e.response.get('Error', {}).get('Code', 'Unknown')
        logger.error(f"S3 delete error: {error_code}", 
                    error=str(e), 
                    bucket=bucket, 
                    object_name=object_name)
        raise