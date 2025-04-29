#!/bin/bash
# Startup script for Pinecone Text Vectorstore App

# Change to the directory where this script is located
cd "$(dirname "$0")"

# Activate virtual environment if it exists
if [ -d "venv" ]; then
  source venv/bin/activate
fi

# Run the Streamlit app
streamlit run app.py 