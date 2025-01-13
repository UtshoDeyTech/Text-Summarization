import os
from fastapi import APIRouter, HTTPException
from fastapi.responses import JSONResponse
from botocore.exceptions import ClientError
from app.service.s3_storage import (
    s3_client, 
    verify_bucket, 
    list_objects,
    S3_BUCKET_NAME,
    S3_END_POINT_URL,
    S3_REGION_NAME,
    S3_ACCESS_KEY,
)
from app.service.log_client import logger

router = APIRouter(prefix="/diagnostics", tags=["Diagnostics"])

@router.get("/s3/detailed")
async def diagnose_s3_detailed():
    """Detailed S3 connection diagnostic"""
    try:
        logger.info("Starting detailed S3 diagnostics")
        
        # Basic configuration check
        logger.info(
            f"Checking S3 configuration | "
            f"bucket={S3_BUCKET_NAME}, "
            f"region={S3_REGION_NAME}, "
            f"has_credentials={bool(S3_ACCESS_KEY) and bool(os.getenv('S3_SECRET_KEY'))}"
        )
        
        diagnostics = {
            "configuration": {
                "bucket_name": S3_BUCKET_NAME,
                "endpoint_url": S3_END_POINT_URL,
                "region": S3_REGION_NAME,
                "access_key_present": bool(S3_ACCESS_KEY),
                "secret_key_present": bool(os.getenv("S3_SECRET_KEY")),
            },
            "tests": {}
        }

        # Test 1: Check bucket existence
        logger.info(f"Testing bucket existence | bucket={S3_BUCKET_NAME}")
        try:
            s3_client.head_bucket(Bucket=S3_BUCKET_NAME)
            logger.info(f"Bucket existence test passed | bucket={S3_BUCKET_NAME}")
            diagnostics["tests"]["bucket_exists"] = {
                "status": "success",
                "message": f"Bucket '{S3_BUCKET_NAME}' exists and is accessible"
            }
        except ClientError as e:
            error_code = e.response.get('Error', {}).get('Code', '0')
            error_message = {
                '403': "Access denied. Check your credentials and bucket permissions.",
                '404': "Bucket does not exist in this region.",
                'NoSuchBucket': "Bucket does not exist.",
            }.get(error_code, f"Unknown error: {str(e)}")
            
            logger.error(f"Bucket existence test failed | bucket={S3_BUCKET_NAME}, error_code={error_code}, error={error_message}")
            diagnostics["tests"]["bucket_exists"] = {
                "status": "error",
                "message": error_message,
                "error_code": error_code
            }

        # Test 2: Try listing objects
        logger.info(f"Testing object listing | bucket={S3_BUCKET_NAME}")
        try:
            response = s3_client.list_objects_v2(
                Bucket=S3_BUCKET_NAME,
                MaxKeys=5
            )
            objects = response.get('Contents', [])
            logger.info(f"Object listing test passed | bucket={S3_BUCKET_NAME}, objects_found={len(objects)}")
            diagnostics["tests"]["list_objects"] = {
                "status": "success",
                "message": f"Successfully listed {len(objects)} objects",
                "sample_objects": [obj['Key'] for obj in objects[:5]] if objects else []
            }
        except ClientError as e:
            error_code = e.response.get('Error', {}).get('Code', 'Unknown')
            logger.error(f"Object listing test failed | bucket={S3_BUCKET_NAME}, error_code={error_code}, error={str(e)}")
            diagnostics["tests"]["list_objects"] = {
                "status": "error",
                "message": f"Error listing objects: {str(e)}",
                "error_code": error_code
            }

        logger.info("S3 diagnostics completed successfully")
        return JSONResponse(content=diagnostics)
        
    except Exception as e:
        error_msg = f"S3 diagnostics failed | error_type={type(e).__name__}, error={str(e)}"
        logger.error(error_msg)
        return JSONResponse(
            status_code=500,
            content={
                "status": "error",
                "message": f"Error in S3 diagnostic: {str(e)}",
                "error_type": type(e).__name__
            }
        )

@router.get("/s3/test-write")
async def test_s3_write():
    """Test S3 write permissions"""
    try:
        logger.info(f"Starting S3 write test | bucket={S3_BUCKET_NAME}")
        
        # Create a small test file
        test_content = b"Hello, this is a test file"
        test_key = f"test/test-file-{os.urandom(4).hex()}.txt"
        
        logger.info(f"Created test file | key={test_key}, size={len(test_content)} bytes")
        
        test_results = {
            "write_test": None,
            "read_test": None,
            "delete_test": None
        }
        
        try:
            # Test write
            logger.info(f"Testing write operation | key={test_key}")
            s3_client.put_object(
                Bucket=S3_BUCKET_NAME,
                Key=test_key,
                Body=test_content
            )
            logger.info(f"Write test successful | key={test_key}")
            test_results["write_test"] = {
                "status": "success",
                "message": "Successfully wrote test file"
            }
            
            # Test read
            logger.info(f"Testing read operation | key={test_key}")
            try:
                response = s3_client.get_object(
                    Bucket=S3_BUCKET_NAME,
                    Key=test_key
                )
                logger.info(f"Read test successful | key={test_key}, size={response['ContentLength']} bytes")
                test_results["read_test"] = {
                    "status": "success",
                    "message": f"Successfully read test file, size: {response['ContentLength']} bytes"
                }
            except ClientError as e:
                error_msg = f"Read test failed | key={test_key}, error={str(e)}"
                logger.error(error_msg)
                test_results["read_test"] = {
                    "status": "error",
                    "message": f"Failed to read test file: {str(e)}"
                }
            
            # Test delete
            logger.info(f"Testing delete operation | key={test_key}")
            try:
                s3_client.delete_object(
                    Bucket=S3_BUCKET_NAME,
                    Key=test_key
                )
                logger.info(f"Delete test successful | key={test_key}")
                test_results["delete_test"] = {
                    "status": "success",
                    "message": "Successfully deleted test file"
                }
            except ClientError as e:
                error_msg = f"Delete test failed | key={test_key}, error={str(e)}"
                logger.error(error_msg)
                test_results["delete_test"] = {
                    "status": "error",
                    "message": f"Failed to delete test file: {str(e)}"
                }
                
        except ClientError as e:
            error_code = e.response.get('Error', {}).get('Code', 'Unknown')
            error_msg = f"Write test failed | key={test_key}, error_code={error_code}, error={str(e)}"
            logger.error(error_msg)
            test_results["write_test"] = {
                "status": "error",
                "message": f"Failed to write test file: {str(e)}",
                "error_code": error_code
            }
        
        logger.info(f"S3 write test completed | all_tests_passed={all(r['status'] == 'success' for r in test_results.values() if r)}")
        return JSONResponse(content={
            "status": "completed",
            "test_file": test_key,
            "results": test_results
        })
            
    except Exception as e:
        error_msg = f"S3 write test failed | error_type={type(e).__name__}, error={str(e)}"
        logger.error(error_msg)
        return JSONResponse(
            status_code=500,
            content={
                "status": "error",
                "message": f"Error in write test: {str(e)}",
                "error_type": type(e).__name__
            }
        )