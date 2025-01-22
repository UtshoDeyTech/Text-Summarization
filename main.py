import os
from dotenv import load_dotenv
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from app.api import list, sync, diagnostics, upload_anc_document, document_upload, document_delete
from app.api.question_answering import qa_router
from app.service.log_client import logger, log_api_request
from time import time

# Load environment variables
load_dotenv()
BACKEND_URL = os.getenv("BACKEND_URL", "askken.io")
DEVELOPMENT_MODE = os.getenv("DEVELOPMENT_MODE", "False").lower() == "true"

# Initial application log
logger.info(f"Application starting | development_mode={DEVELOPMENT_MODE}, backend_url={BACKEND_URL}")

# Initialize FastAPI app with conditional UI settings
app = FastAPI(
    docs_url="/docs" if DEVELOPMENT_MODE else None,
    redoc_url="/redoc" if DEVELOPMENT_MODE else None,
    openapi_url="/openapi.json" if DEVELOPMENT_MODE else None,
)


# Configure CORS
origins = [
    "http://localhost:3000",           
    "http://localhost:8000",           
    "*"
]

logger.info(f"Configuring CORS | origins={origins}")

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"]
)

# Include routers
app.include_router(document_upload.router)
app.include_router(upload_anc_document.router)
app.include_router(document_delete.router)
app.include_router(list.router)
app.include_router(sync.router)
app.include_router(qa_router.router)
app.include_router(diagnostics.router)

@app.middleware("http")
async def log_requests(request: Request, call_next):
    start_time = time()
    response = await call_next(request)
    duration = time() - start_time
    
    # Log request details
    log_api_request(
        endpoint=str(request.url.path),
        method=request.method,
        status_code=response.status_code,
        duration_ms=round(duration * 1000, 2),
        client_host=request.client.host if request.client else None,
        query_params=dict(request.query_params)
    )
    
    return response

@app.get("/")
async def root():
    log_api_request("/", "GET")
    return {
        "message": "PDF Processing API is running",
        "version": "1.0.0",
        "docs": "/docs" if DEVELOPMENT_MODE else "Not available in production",
        "diagnostics": "/diagnostics/system"
    }

@app.on_event("startup")
async def startup_event():
    routes = [f"{route.methods} {route.path}" for route in app.routes]
    logger.info(
        f"Application started | "
        f"backend_url={BACKEND_URL}, "
        f"routes={routes}, "
        f"development_mode={DEVELOPMENT_MODE}"
    )

@app.on_event("shutdown")
async def shutdown_event():
    logger.info("Application shutting down")

if __name__ == "__main__":
    import uvicorn
    logger.info("Starting FastAPI server")
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8000,
        log_config=None  # Disable uvicorn's default logging
    )