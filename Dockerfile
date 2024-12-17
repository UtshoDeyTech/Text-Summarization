# Use an official Python runtime as the base image
FROM python:3.9-slim

# Set the working directory in the container
WORKDIR /app

# Install necessary system dependencies for AWS and Playwright
RUN apt-get update && apt-get install -y \
    wget \
    gnupg \
    libglib2.0-0 \
    libgobject-2.0-0 \
    libnss3 \
    libnssutil3 \
    libnspr4 \
    libdbus-1-3 \
    libatk1.0-0 \
    libatk-bridge2.0-0 \
    libcups2 \
    libdrm2 \
    libgio-2.0-0 \
    libexpat1 \
    libatspi2.0-0 \
    libx11-6 \
    libxcomposite1 \
    libxdamage1 \
    libxext6 \
    libxfixes3 \
    libxrandr2 \
    libgbm1 \
    libxcb1 \
    libxkbcommon0 \
    libasound2 \
    libpango-1.0-0 \
    libcairo2 \
    fonts-liberation \
    xvfb \
    && rm -rf /var/lib/apt/lists/*

# Copy the requirements file into the container
COPY requirements.txt .

# Install Python packages
RUN pip install --no-cache-dir -r requirements.txt

# Install Playwright with all dependencies
RUN DEBIAN_FRONTEND=noninteractive playwright install-deps
RUN playwright install chromium

# Copy the application code
COPY . .

# Create a non-root user
RUN useradd -m myuser
RUN chown -R myuser:myuser /app
USER myuser

# Expose the port
EXPOSE 8000

# Start Xvfb and run the FastAPI application
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]