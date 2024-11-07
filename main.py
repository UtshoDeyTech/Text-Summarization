import logging
import os
from dotenv import load_dotenv
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from app.api import upload, delete, list, search, sync, qa, diagnostics
from app.service.log_client import logger, log_api_request, log_error
from time import time

# Load environment variables
load_dotenv()
BACKEND_URL = os.getenv("BACKEND_URL", "askken.io")
DEVELOPMENT_MODE = os.getenv("DEVELOPMENT_MODE", "False").lower() == "true"

logger.info(f"Application starting | development_mode={DEVELOPMENT_MODE}, backend_url={BACKEND_URL}")

app = FastAPI(
    docs_url="/docs" if DEVELOPMENT_MODE else None,
    redoc_url="/redoc" if DEVELOPMENT_MODE else None,
    openapi_url="/openapi.json" if DEVELOPMENT_MODE else None,
)

def get_allowed_origins():
    # Development origins
    origins = [
        "http://localhost:3000",
        "http://localhost:8000",
    ]
    
    # Production origins with wildcard subdomain support
    if BACKEND_URL:
        # Add both HTTP and HTTPS for the main domain and all subdomains
        origins.extend([
            f"https://*.{BACKEND_URL}",  # Allow all HTTPS subdomains
            f"http://*.{BACKEND_URL}",   # Allow all HTTP subdomains
            f"https://{BACKEND_URL}",    # Allow main domain with HTTPS
            f"http://{BACKEND_URL}",     # Allow main domain with HTTP
        ])
    
    logger.info(f"Configured CORS origins | origins={origins}")
    return origins

app.add_middleware(
    CORSMiddleware,
    allow_origins=get_allowed_origins(),
    allow_origin_regex=f"https?://.*\.{BACKEND_URL.replace('.', '\.')}$",  # Regex for dynamic subdomain matching
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
    expose_headers=["*"],
    max_age=3600,  # Cache preflight requests for 1 hour
)

# Include routers
app.include_router(upload.router)
app.include_router(delete.router)
app.include_router(list.router)
app.include_router(search.router)
app.include_router(sync.router)
app.include_router(qa.router)
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