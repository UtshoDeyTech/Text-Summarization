# Use an official Python runtime as a parent image
FROM python:3.9-slim as builder

# Set the working directory in the container
WORKDIR /app

# Copy the current directory contents into the container
COPY . /app

# Create a Python script for retrying pip install
RUN echo 'import sys\n\
from retrying import retry\n\
from pip._internal.cli.main import main\n\
\n\
@retry(stop_max_attempt_number=5, wait_fixed=10000)\n\
def install_requirements():\n\
    sys.exit(main(["install", "--no-cache-dir", "-r", "requirements.txt", "--timeout", "100"]))\n\
\n\
if __name__ == "__main__":\n\
    install_requirements()' > retry_install.py

# Install any needed packages specified in requirements.txt
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir retrying && \
    python retry_install.py

# Start a new stage for a smaller final image
FROM python:3.9-slim

# Set the working directory in the container
WORKDIR /app

# Copy the installed packages from the builder stage
COPY --from=builder /usr/local/lib/python3.9/site-packages /usr/local/lib/python3.9/site-packages

# Copy the application from the builder stage
COPY --from=builder /app /app

# Make port 5000 available to the world outside this container
EXPOSE 5000

# Define environment variable
ENV NAME PDFChatbot

# Run app.py when the container launches
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "5000"]