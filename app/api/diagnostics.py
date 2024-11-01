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

router = APIRouter(prefix="/diagnostics", tags=["diagnostics"])

@router.get("/s3/detailed")
async def diagnose_s3_detailed():
    """Detailed S3 connection diagnostic"""
    try:
        # Basic configuration check
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
        try:
            s3_client.head_bucket(Bucket=S3_BUCKET_NAME)
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
            
            diagnostics["tests"]["bucket_exists"] = {
                "status": "error",
                "message": error_message,
                "error_code": error_code
            }

        # Test 2: Try listing objects
        try:
            response = s3_client.list_objects_v2(
                Bucket=S3_BUCKET_NAME,
                MaxKeys=5
            )
            objects = response.get('Contents', [])
            diagnostics["tests"]["list_objects"] = {
                "status": "success",
                "message": f"Successfully listed {len(objects)} objects",
                "sample_objects": [obj['Key'] for obj in objects[:5]] if objects else []
            }
        except ClientError as e:
            diagnostics["tests"]["list_objects"] = {
                "status": "error",
                "message": f"Error listing objects: {str(e)}",
                "error_code": e.response.get('Error', {}).get('Code', 'Unknown')
            }

        return JSONResponse(content=diagnostics)
        
    except Exception as e:
        logger.error(
            "Error in detailed S3 diagnostic",
            error=str(e),
            error_type=type(e).__name__
        )
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
        # Create a small test file
        test_content = b"Hello, this is a test file"
        test_key = f"test/test-file-{os.urandom(4).hex()}.txt"
        
        test_results = {
            "write_test": None,
            "read_test": None,
            "delete_test": None
        }
        
        try:
            # Test write
            s3_client.put_object(
                Bucket=S3_BUCKET_NAME,
                Key=test_key,
                Body=test_content
            )
            test_results["write_test"] = {
                "status": "success",
                "message": "Successfully wrote test file"
            }
            
            # Test read
            try:
                response = s3_client.get_object(
                    Bucket=S3_BUCKET_NAME,
                    Key=test_key
                )
                test_results["read_test"] = {
                    "status": "success",
                    "message": f"Successfully read test file, size: {response['ContentLength']} bytes"
                }
            except ClientError as e:
                test_results["read_test"] = {
                    "status": "error",
                    "message": f"Failed to read test file: {str(e)}"
                }
            
            # Test delete
            try:
                s3_client.delete_object(
                    Bucket=S3_BUCKET_NAME,
                    Key=test_key
                )
                test_results["delete_test"] = {
                    "status": "success",
                    "message": "Successfully deleted test file"
                }
            except ClientError as e:
                test_results["delete_test"] = {
                    "status": "error",
                    "message": f"Failed to delete test file: {str(e)}"
                }
                
        except ClientError as e:
            error_code = e.response.get('Error', {}).get('Code', 'Unknown')
            test_results["write_test"] = {
                "status": "error",
                "message": f"Failed to write test file: {str(e)}",
                "error_code": error_code
            }
        
        return JSONResponse(content={
            "status": "completed",
            "test_file": test_key,
            "results": test_results
        })
            
    except Exception as e:
        logger.error(
            "Error in S3 write test",
            error=str(e),
            error_type=type(e).__name__
        )
        return JSONResponse(
            status_code=500,
            content={
                "status": "error",
                "message": f"Error in write test: {str(e)}",
                "error_type": type(e).__name__
            }
        )

# @router.get("/system")
# async def system_diagnostics():
#     """General system diagnostic information"""
#     try:
#         # Check environment variables
#         env_vars = {
#             "DEVELOPMENT_MODE": os.getenv("DEVELOPMENT_MODE", "False"),
#             "BACKEND_URL": os.getenv("BACKEND_URL", "Not set"),
#             "S3_BUCKET_NAME": S3_BUCKET_NAME,
#             "S3_REGION_NAME": S3_REGION_NAME,
#             "S3_END_POINT_URL": S3_END_POINT_URL,
#             "S3_ACCESS_KEY_PRESENT": bool(S3_ACCESS_KEY),
#             "S3_SECRET_KEY_PRESENT": bool(os.getenv("S3_SECRET_KEY"))
#         }
        
#         # Check S3 connectivity
#         s3_status = "unknown"
#         try:
#             s3_client.head_bucket(Bucket=S3_BUCKET_NAME)
#             s3_status = "connected"
#         except ClientError as e:
#             error_code = e.response.get('Error', {}).get('Code', 'Unknown')
#             s3_status = f"error: {error_code}"
#         except Exception as e:
#             s3_status = f"error: {str(e)}"
        
#         return {
#             "status": "success",
#             "environment": env_vars,
#             "service_status": {
#                 "s3": s3_status
#             }
#         }
#     except Exception as e:
#         logger.error(
#             "Error in system diagnostic",
#             error=str(e),
#             error_type=type(e).__name__
#         )
#         raise HTTPException(
#             status_code=500,
#             detail=f"Error running system diagnostics: {str(e)}"
#         )