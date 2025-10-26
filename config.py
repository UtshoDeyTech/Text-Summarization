import os
from dotenv import load_dotenv

load_dotenv()
DEVELOPMENT_MODE=os.environ.get("DEVELOPMENT_MODE")

OPENAI_API_KEY=os.environ.get("OPENAI_API_KEY")

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

DB_HOST=os.environ.get("DB_HOST")
DB_USER=os.environ.get("DB_USER")
DB_PASSWORD=os.environ.get("DB_PASSWORD")
DB_NAME=os.environ.get("DB_NAME")
DB_PORT=os.environ.get("DB_PORT")

MAX_CHUNKS = 5
MODEL = "gpt-3.5-turbo"
NUM_SUGGESTIONS = 3

PERPLEXITY_API_KEY= os.environ.get("PERPLEXITY_API_KEY")  # Default key for testing

# Qdrant Configuration
QDRANT_URL = os.environ.get("QDRANT_URL", "http://localhost:6333")  # Qdrant connection URL
QDRANT_API_KEY = os.environ.get("QDRANT_API_KEY", "")  # Optional Qdrant native API key
QDRANT_COLLECTION_NAME = os.environ.get("QDRANT_COLLECTION_NAME", "documents")  # Collection name
QDRANT_TIMEOUT = int(os.environ.get("QDRANT_TIMEOUT", "60"))  # Connection timeout in seconds

# HTTP Basic Auth for Qdrant (when using Nginx reverse proxy)
QDRANT_USERNAME = os.environ.get("QDRANT_USERNAME", "")  # Username for HTTP Basic Auth
QDRANT_PASSWORD = os.environ.get("QDRANT_PASSWORD", "")  # Password for HTTP Basic Auth