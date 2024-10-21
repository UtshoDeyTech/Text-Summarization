# PDF Processing API For ASKEN.io

This project is a FastAPI-based API for processing and searching PDF documents using OpenAI embeddings and Pinecone vector database.

## Prerequisites

- Python 3.8+
- pip (Python package manager)
- An OpenAI API key
- A Pinecone API key (free tier is sufficient)
- An S3-compatible storage service (e.g., AWS S3, MinIO)

## Setup

1. Clone the repository:
   ```
   git clone <repo_name>
   cd <repo_name>
   ```

2. Create a virtual environment:
   ```
   python -m venv env
   ```

3. Activate the virtual environment:
   - On Windows:
     ```
     .\env\Scripts\activate
     ```
   - On macOS and Linux:
     ```
     source env/bin/activate
     ```

4. Install the required packages:
   ```
   pip install -r requirements.txt
   ```

5. Create a `.env` file in the root directory with the following content:
   ```
   OPENAI_API_KEY=your_openai_api_key
   PINECONE_API_KEY=your_pinecone_api_key
   PINECONE_ENVIRONMENT=gcp-starter
   S3_REGION_NAME=your_s3_region
   S3_END_POINT_URL=your_s3_endpoint_url
   S3_ACCESS_KEY=your_s3_access_key
   S3_SECRET_KEY=your_s3_secret_key
   S3_BUCKET_NAME=your_s3_bucket_name
   ```
   Replace the placeholder values with your actual API keys and configuration.

## Running the Application

1. Start the FastAPI server:
   ```
   uvicorn main:app --host 0.0.0.0 --port 8000 --reload
   ```

2. The API will be available at `http://localhost:8000`

3. Access the OpenAPI documentation at `http://localhost:8000/docs`

## API Endpoints

- `POST /upload_pdf`: Upload a PDF file
- `DELETE /delete_pdf/{pdf_id}`: Delete a PDF and its associated vectors
- `GET /list_pdfs`: List all uploaded PDFs
- `POST /search_chunks`: Search for relevant chunks across all PDFs
- `GET /search_pdf/{pdf_id}`: Search within a specific PDF
- `POST /sync_pinecone`: Synchronize Pinecone with S3 storage


## Troubleshooting

If you encounter any issues:

1. Ensure all environment variables are correctly set in the `.env` file.
2. Check the console output for any error messages.
3. Verify that your OpenAI and Pinecone API keys are valid and have the necessary permissions.
4. Make sure your S3-compatible storage service is properly configured and accessible.

#