# Use an official Python runtime as the base image
FROM python:3.9-slim

# Set environment variables
ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PATH="/home/app/.local/bin:${PATH}" \
    VIRTUAL_ENV=/opt/venv

# Create a non-root user
RUN useradd --create-home app \
    && mkdir -p /app \
    && chown -R app:app /app

# Create and activate virtual environment
RUN python -m venv $VIRTUAL_ENV
ENV PATH="$VIRTUAL_ENV/bin:$PATH"

# Set the working directory in the container
WORKDIR /app

# Install system dependencies for Playwright
RUN apt-get update && apt-get install -y \
    libglib2.0-0 \
    libnss3 \
    libnspr4 \
    libdbus-1-3 \
    libatk1.0-0 \
    libatk-bridge2.0-0 \
    libxcomposite1 \
    libxdamage1 \
    libxext6 \
    libxfixes3 \
    libxrandr2 \
    libgbm1 \
    libdrm2 \
    libxcb1 \
    libxkbcommon0 \
    libasound2 \
    libatspi2.0-0 \
    libx11-6 \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Switch to non-root user
USER app

# Upgrade pip and install dependencies in virtual environment
RUN python -m pip install --upgrade pip setuptools wheel

# Copy requirements with correct ownership
COPY --chown=app:app requirements.txt .

# Install dependencies in the virtual environment
RUN pip install urllib3>=1.25.4,<1.27 && \
    pip install -r requirements.txt

# Install Playwright browser with dependencies
RUN playwright install chromium && \
    playwright install-deps

# Copy the application code with correct ownership
COPY --chown=app:app . .

# Verify uvicorn installation
RUN which uvicorn && pip list

# Expose the port the app will run on
EXPOSE 8000

# Run FastAPI with uvicorn
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]