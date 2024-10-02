# Running PDF Chatbot with Docker

This guide will walk you through the process of running the PDF Chatbot application using Docker and Docker Compose.

## Prerequisites

- Docker (with Compose V2)

## Steps

1. Ensure you are in the project root directory.

2. Create a `.env` file in the project root with your API keys:
   ```
   OPENAI_API_KEY=your_openai_api_key_here
   PINECONE_API_KEY=your_pinecone_api_key_here
   ```

3. Build the Docker images:
   ```
   docker compose build
   ```

4. Start the services:
   ```
   docker compose up
   ```

5. Access the applications:
   - FastAPI backend: http://localhost:5000
   - Streamlit frontend: http://localhost:8501

6. To stop the services, press `Ctrl+C` in the terminal where Docker Compose is running, or run:
   ```
   docker compose down
   ```

## Notes

- The `Files` folder in the project root is mounted as a volume in the containers. Any PDFs you add to this folder will be accessible to the application.
- Changes to the Python code will trigger a reload of the FastAPI server due to the `--reload` flag in the backend service command.
- If you make changes to the Dockerfile or need to rebuild the images, use `docker compose build` before running `docker compose up`.

## Troubleshooting

- If you encounter issues with permissions for the `Files` folder, ensure that the folder exists and has the correct permissions:
  ```
  mkdir -p Files
  chmod 777 Files
  ```

- If the services fail to start, check the Docker logs for error messages:
  ```
  docker compose logs
  ```

- Ensure that your `.env` file is properly formatted and contains the correct API keys.

- If you need to rebuild the images after making changes, use:
  ```
  docker compose build --no-cache
  ```

- If you're using an older version of Docker that doesn't support Compose V2, you may need to use `docker-compose` (with a hyphen) instead of `docker compose`. In this case, replace `docker compose` with `docker-compose` in all the commands above.

Remember to never commit your `.env` file or share it publicly, as it contains sensitive API keys.