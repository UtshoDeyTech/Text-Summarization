#!/bin/bash

# Check if virtual environment exists, create if not
if [ ! -d "env" ]; then
    echo "Creating virtual environment..."
    python3 -m venv env
else
    echo "Virtual environment already exists."
fi

# Activate virtual environment
source env/bin/activate

# Upgrade pip and install dependencies
pip install --upgrade pip
pip install -r requirements.txt

# Create .env file if it doesn't exist and add OPENAI_API_KEY line
if [ ! -f ".env" ]; then
    echo "Creating .env file..."
    touch .env
    echo "OPENAI_API_KEY=" >> .env
else
    echo ".env file already exists."
fi
