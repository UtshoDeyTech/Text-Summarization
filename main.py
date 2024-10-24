import logging
from fastapi import FastAPI
from app.api import upload, delete, list, search, sync, qa
from app.service.log_client import logger

# FastAPI app
app = FastAPI()


# Include routers
app.include_router(upload.router)
app.include_router(delete.router)
app.include_router(list.router)
app.include_router(search.router)
app.include_router(sync.router)
app.include_router(qa.router)

@app.get("/")
async def root():
    return {"message": "PDF Processing API is running"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)