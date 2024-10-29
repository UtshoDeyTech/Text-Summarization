import logging
import os
from dotenv import load_dotenv
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from app.api import upload, delete, list, search, sync, qa
from app.service.log_client import logger

# Load environment variables
load_dotenv()
BACKEND_URL = os.getenv("BACKEND_URL", "askken.io")
DEVELOPMENT_MODE = os.getenv("DEVELOPMENT_MODE", "False").lower() == "true"

logger.info(f"Development mode: {DEVELOPMENT_MODE}")

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
    f"https://{BACKEND_URL}",          
    f"https://www.{BACKEND_URL}",      
    f"https://api.{BACKEND_URL}",      
    f"http://{BACKEND_URL}",           
    f"http://www.{BACKEND_URL}",       
    f"http://api.{BACKEND_URL}"        
]

logger.info(f"Configured CORS origins: {origins}")

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"]
)

# Include routers
app.include_router(upload.router)
app.include_router(delete.router)
app.include_router(list.router)
app.include_router(search.router)
app.include_router(sync.router)
app.include_router(qa.router)

@app.get("/")
async def root():
    logger.info("Root endpoint accessed")
    return {
        "message": "PDF Processing API is running",
        "version": "1.0.0",
        "docs": f"/docs" if DEVELOPMENT_MODE else "Not available in production"
    }

@app.on_event("startup")
async def startup_event():
    logger.info(f"Backend URL: {BACKEND_URL}")
    logger.info("Available endpoints:")
    for route in app.routes:
        logger.info(f"  {route.methods}{route.path}")

if __name__ == "__main__":
    import uvicorn
    logger.info("Starting FastAPI server")
    uvicorn.run(app, host="0.0.0.0", port=8000)