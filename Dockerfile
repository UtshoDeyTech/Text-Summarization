# Stage 1: Build stage
FROM python:3.9-slim AS builder

# Set the working directory
WORKDIR /app

# Install system dependencies for Playwright and Chromium
RUN apt-get update && apt-get install -y \
    wget \
    unzip \
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
    && rm -rf /var/lib/apt/lists/*

# Download and install the headless Chromium shell
RUN wget -q -O chromium-headless-shell.zip https://playwright.azureedge.net/builds/chromium/1148/chromium-headless-shell-linux.zip && \
    unzip chromium-headless-shell.zip -d /opt/chromium-headless-shell && \
    rm chromium-headless-shell.zip && \
    ln -s /opt/chromium-headless-shell/chromium-headless-shell /usr/bin/chromium

# Copy the requirements file into the container
COPY requirements.txt .

# Upgrade pip and install Python dependencies
RUN pip install --upgrade pip && \
    pip install --no-cache-dir --use-deprecated=legacy-resolver -r requirements.txt

# Copy the rest of the application code
COPY . .

# Stage 2: Final stage
FROM python:3.9-slim

# Set the working directory
WORKDIR /app

# Copy only the necessary files from the builder stage
COPY --from=builder /app /app
COPY --from=builder /opt/chromium-headless-shell /opt/chromium-headless-shell

# Recreate the symbolic link in the final stage
RUN ln -s /opt/chromium-headless-shell/chromium-headless-shell /usr/bin/chromium

# Copy the Python packages from the builder stage
COPY --from=builder /usr/local/lib/python3.9/site-packages /usr/local/lib/python3.9/site-packages
COPY --from=builder /usr/local/bin /usr/local/bin

# Verify uvicorn is installed and in PATH
RUN echo $PATH && which uvicorn && pip list

# Expose the port the app will run on
EXPOSE 8000

# Run FastAPI with uvicorn
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]