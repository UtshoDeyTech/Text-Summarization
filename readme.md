# PDF Chatbot

This application is a PDF Chatbot that allows users to upload PDFs, query their content, and receive AI-generated responses based on the PDF contents.

## Prerequisites

- Python 3.8+
- Docker
- pip (Python package manager)

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

4. Set up ChromaDB using Docker:
   ```
   docker pull chromadb/chroma
   docker volume create chroma-data
   docker run -d -p 8000:8000 -v chroma-data:/chroma/chroma chromadb/chroma
   ```

   Note the container ID from the output of the `docker run` command.

5. Create a `.env` file in the project root and add your OpenAI API key:
   ```
   OPENAI_API_KEY=your_api_key_here
   ```

## Running the Application

1. Start the ChromaDB container (if it's not already running):
   ```
   docker start <container-id>
   ```

2. Start the FastAPI backend:
   ```
   uvicorn app:app --reload --port 5000
   ```

3. In a new terminal, start the Streamlit frontend:
   ```
   streamlit run chatbot.py
   ```

4. Open your web browser and go to `http://localhost:8501` to access the PDF Chatbot interface.

## Usage

1. Use the FastAPI backend to upload PDF files. You can do this through the FastAPI Swagger UI at `http://localhost:5000/docs`.

2. Once PDFs are uploaded, you can ask questions about their content using the Streamlit interface.

3. The application will search for relevant information in the PDFs and attempt to generate an AI response using the OpenAI API.

4. If there are issues with the OpenAI API, the application will fall back to providing the most relevant text chunk from the PDFs.

## Troubleshooting

- If you encounter issues with ChromaDB, ensure the Docker container is running:
  ```
  docker ps
  ```
  If it's not listed, start it using the command from step 1 in the "Running the Application" section.

- If you face OpenAI API errors, check your API key in the `.env` file and ensure you have sufficient quota.

- For any other issues, check the console outputs of both the FastAPI backend and Streamlit frontend for error messages.

## Notes

- The `Files` folder in the project root is used to store uploaded PDFs.
- The application uses a local ChromaDB instance for vector storage and similarity search.
- The OpenAI model used is "gpt-4o-mini". Ensure this model is available in your OpenAI account.

