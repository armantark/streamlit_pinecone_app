#!/bin/bash
# Setup script for Pinecone Text Vectorstore App

# Change to the directory where this script is located
cd "$(dirname "$0")"

# Create virtual environment if it doesn't exist
if [ ! -d "venv" ]; then
  python3 -m venv venv
fi

# Activate virtual environment
source venv/bin/activate

# Upgrade pip
pip install --upgrade pip

# Install requirements
pip install -r requirements.txt

echo "Setup complete. You can now run the app with: bash startup.sh" 