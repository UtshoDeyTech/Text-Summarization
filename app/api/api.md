# PDF Question-Answering System API Documentation

## Base URL
```
http://<your-domain>/
```

## Authentication
All endpoints require appropriate authentication headers (to be implemented based on your authentication system).

## Endpoints

### 1. Upload PDF
Upload a PDF document for processing and vectorization.

**Endpoint:** `POST /upload_pdf`  
**Content-Type:** `multipart/form-data`

**Request Body:**
```
file: PDF file (required)
```

**Response:**
```json
{
    "pdf_id": "string (UUID)",
    "chunks_stored": "integer",
    "s3_url": "string"
}
```

**Error Responses:**
- `400`: Invalid file format (non-PDF)
- `500`: Upload/processing failure

### 2. Question Answering
Ask questions about uploaded documents and get AI-generated answers.

**Endpoint:** `POST /ask`  
**Content-Type:** `application/json`

**Request Body:**
```json
{
    "question": "string (required)",
    "max_chunks": "integer (default: 3)",
    "model": "string (gpt-3.5-turbo or gpt-4, default: gpt-3.5-turbo)",
    "num_suggestions": "integer (1-10, default: 3)"
}
```

**Response:**
```json
{
    "question": "string",
    "answer": "string",
    "chunks": [
        {
            "text": "string",
            "pdf_id": "string",
            "score": "float"
        }
    ],
    "suggested_questions": [
        "string"
    ],
    "model_used": "string"
}
```

**Error Responses:**
- `400`: Invalid request parameters
- `500`: Processing error

### 3. List PDFs
Get a list of all uploaded PDF documents.

**Endpoint:** `GET /list_pdfs`

**Response:**
```json
{
    "total_pdfs": "integer",
    "pdfs": [
        {
            "id": "string",
            "name": "string"
        }
    ]
}
```

**Error Response:**
- `500`: Server error

### 4. Delete PDF
Delete a PDF and its associated vector embeddings.

**Endpoint:** `DELETE /delete_pdf/{pdf_id}`

**Parameters:**
- `pdf_id`: UUID of the PDF to delete (path parameter)

**Response:**
```json
{
    "message": "string"
}
```

**Error Responses:**
- `404`: PDF not found
- `500`: Deletion error

### 5. Sync Vectors
Synchronize vector database with stored PDFs.

**Endpoint:** `POST /sync_pinecone`

**Response:**
```json
{
    "message": "string",
    "deleted_pdfs": [
        "string"
    ]
}
```

**Error Response:**
- `500`: Synchronization error

## Response Status Codes
- `200`: Successful operation
- `400`: Bad request (invalid input)
- `404`: Resource not found
- `500`: Server error

## Common Error Response Format
```json
{
    "detail": "Error message description"
}
```

## Rate Limiting
- Default: 100 requests per minute per IP
- Upload endpoints: 10 requests per minute per IP
- Question answering endpoints: 50 requests per minute per IP

## File Size Limits
- Maximum PDF size: 10MB
- Maximum number of pages: 100

## Notes
1. All endpoints return JSON responses
2. Timestamps are in ISO 8601 format
3. PDF IDs are UUIDs
4. Vector similarity scores range from 0 to 1
5. The question answering system uses semantic search to find relevant content

## Best Practices
1. **Upload PDFs:**
   - Ensure PDFs are text-searchable
   - Keep file sizes under 10MB
   - Use meaningful filenames

2. **Asking Questions:**
   - Be specific in your questions
   - Start with a lower `max_chunks` value (3-5)
   - Use GPT-4 for complex queries
   - Adjust `num_suggestions` based on content complexity

3. **Error Handling:**
   - Implement proper retry logic
   - Handle rate limiting gracefully
   - Log error responses for debugging

4. **Performance:**
   - Cache frequently accessed PDFs
   - Use batch operations when possible
   - Monitor response times and adjust `max_chunks` accordingly

