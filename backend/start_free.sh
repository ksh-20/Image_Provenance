#!/bin/bash

echo "Starting Free Deepfake Detection Backend..."
echo ""

# Check if virtual environment exists
if [ ! -d "venv" ]; then
    echo "Creating virtual environment..."
    python3 -m venv venv
fi

# Activate virtual environment
source venv/bin/activate

# Install requirements
echo "Installing dependencies..."
pip install -r requirements_free.txt

# Start the server
echo ""
echo "Starting API server on http://localhost:5000"
echo ""
python deepfake_free.py

