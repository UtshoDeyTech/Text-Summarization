# PDF Chatbot

This application is a PDF Chatbot that allows users to upload PDFs, query their content, and receive AI-generated responses based on the PDF contents. It uses Pinecone for vector storage and similarity search.

## Prerequisites

- Python 3.8+
- pip (Python package manager)
- Pinecone account (for vector database)
- OpenAI API key

## Setup

1. Clone the repository:
   ```
   git clone <repository-url>
   cd <repository-directory>
   ```

2. Create a virtual environment (optional but recommended):
   ```
   python -m venv venv
   source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
   ```

3. Install the required packages:
   ```
   pip install -r requirements.txt
   ```

4. Set up a Pinecone account and create an index:
   - Sign up at [https://www.pinecone.io/](https://www.pinecone.io/)
   - Create a new project and note your API key
   - Create an index with the following settings:
     - Dimensions: 1536 (for OpenAI's text-embedding-ada-002 model)
     - Metric: Cosine
     - Pod Type: Serverless

5. Create a `.env` file in the project root and add your API keys:
   ```
   OPENAI_API_KEY=your_openai_api_key_here
   PINECONE_API_KEY=your_pinecone_api_key_here
   ```

## Running the Application

1. Start the FastAPI backend:
   ```
   uvicorn app:app --reload --port 5000
   ```

2. In a new terminal, start the Streamlit frontend:
   ```
   streamlit run chatbot.py
   ```

3. Open your web browser and go to `http://localhost:8501` to access the PDF Chatbot interface.

## Usage

1. Use the FastAPI backend to upload PDF files. You can do this through the FastAPI Swagger UI at `http://localhost:5000/docs`.

2. Once PDFs are uploaded, you can ask questions about their content using the Streamlit interface.

3. The application will search for relevant information in the PDFs using Pinecone and attempt to generate an AI response using the OpenAI API.

4. If there are issues with the OpenAI API, the application will fall back to providing the most relevant text chunk from the PDFs.

5. You can use the "Sync Pinecone with PDF folder" button in the Streamlit interface to ensure Pinecone is up-to-date with the PDFs in your local folder.

## Troubleshooting

- If you encounter issues with Pinecone, check your API key in the `.env` file and ensure your index is set up correctly.

- If you face OpenAI API errors, check your API key in the `.env` file and ensure you have sufficient quota.

- If PDFs are not being deleted from Pinecone when deleted from the local folder, use the "Sync Pinecone with PDF folder" button in the Streamlit interface.

- For any other issues, check the console outputs of both the FastAPI backend and Streamlit frontend for error messages.

## Notes

- The `Files` folder in the project root is used to store uploaded PDFs.
- The application uses Pinecone for vector storage and similarity search.
- The OpenAI model used for embeddings is "text-embedding-ada-002".
- The OpenAI model used for chat completions is "gpt-4o-mini". Ensure this model is available in your OpenAI account.

## Future Improvements

- Implement user authentication and multi-user support.
- Add support for more file types beyond PDFs.
- Improve error handling and user feedback.
- Implement caching to reduce API calls and improve performance.