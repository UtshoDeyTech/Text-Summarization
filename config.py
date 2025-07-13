import os
from dotenv import load_dotenv

load_dotenv()
DEVELOPMENT_MODE=os.environ.get("DEVELOPMENT_MODE")

OPENAI_API_KEY=os.environ.get("OPENAI_API_KEY")

PINECONE_API_KEY=os.environ.get("PINECONE_API_KEY")
PINECONE_ENVIRONMENT=os.environ.get("PINECONE_ENVIRONMENT")

PINECONE_CLIENT_INDEX=os.environ.get("PINECONE_CLIENT_INDEX")
PINECONE_ANC_INDEX=os.environ.get("PINECONE_ANC_INDEX")
PINECONE_GLOBAL_INDEX=os.environ.get("PINECONE_GLOBAL_INDEX")

S3_REGION_NAME=os.environ.get("S3_REGION_NAME")
S3_END_POINT_URL=os.environ.get("S3_END_POINT_URL")
S3_ACCESS_KEY=os.environ.get("S3_ACCESS_KEY")
S3_SECRET_KEY=os.environ.get("S3_SECRET_KEY")
S3_BUCKET_NAME=os.environ.get("S3_BUCKET_NAME")

BACKEND_URL=os.environ.get("BACKEND_URL")

SEQ_URL = os.getenv("SEQ_URL")
SEQ_API_KEY = os.getenv("SEQ_API_KEY")
LOG_ENVIROMENT=os.environ.get("LOG_ENVIROMENT")
LOG_APP_NAME=os.environ.get("LOG_APP_NAME")

AI_VALUE_ASP=os.environ.get("AI_VALUE_ASP")

VALIDATION_DAYS=os.environ.get("VALIDATION_DAYS")

LLAMA_URL="https://ollama.askken.ai/"

MAX_CHUNKS = 5
MODEL = "gpt-3.5-turbo"
NUM_SUGGESTIONS = 3

PERPLEXITY_API_KEY= os.environ.get("PERPLEXITY_API_KEY")  # Default key for testing