import os
import logging
import boto3
from botocore.exceptions import ClientError

logger = logging.getLogger(__name__)

S3_REGION_NAME = os.getenv("S3_REGION_NAME")
S3_END_POINT_URL = os.getenv("S3_END_POINT_URL")
S3_ACCESS_KEY = os.getenv("S3_ACCESS_KEY")
S3_SECRET_KEY = os.getenv("S3_SECRET_KEY")
S3_BUCKET_NAME = os.getenv("S3_BUCKET_NAME")

s3_client = boto3.client(
    "s3",
    region_name=S3_REGION_NAME,
    endpoint_url=S3_END_POINT_URL,
    aws_access_key_id=S3_ACCESS_KEY,
    aws_secret_access_key=S3_SECRET_KEY,
)

def upload_file(file_obj, bucket, object_name, expiration=3600):
    try:
        s3_client.upload_fileobj(file_obj, bucket, object_name)
    except ClientError as e:
        logger.error(e)
        return None

    try:
        response = s3_client.generate_presigned_url(
            "get_object",
            Params={"Bucket": bucket, "Key": object_name},
            ExpiresIn=expiration,
        )
    except ClientError as e:
        logger.error(e)
        return None

    return response

def list_objects(bucket):
    try:
        return s3_client.list_objects_v2(Bucket=bucket)
    except ClientError as e:
        logger.error(e)
        raise

def delete_object(bucket, object_name):
    try:
        s3_client.delete_object(Bucket=bucket, Key=object_name)
    except ClientError as e:
        logger.error(e)
        raise