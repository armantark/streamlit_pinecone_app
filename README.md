# Pinecone Text Vectorstore App

A Streamlit application for searching similar texts and adding new content to a Pinecone vector database using OpenAI embeddings.

## Features

- **Semantic Search:** Find semantically similar texts to your queries using OpenAI embeddings and Pinecone
- **Insert Text:** Add new text documents with custom metadata to your vector database
- **User-Friendly Interface:** Modern UI with tabs, status indicators, and responsive design

## Installation

1. Install the dependencies:
   ```
   pip install -r requirements.txt
   ```

2. Ensure you have a `.env` file in the project root directory with your Pinecone and OpenAI API credentials.

## Setup

Before running the app, set up your Python environment and install dependencies:

From the project root directory, run:

```bash
bash setup.sh
```

This will:
- Create a Python virtual environment (if it doesn't exist)
- Activate the virtual environment
- Install all required dependencies from requirements.txt

## Running the App

From the project root directory, run:

```bash
streamlit run app.py
```

## Quick Start with startup.sh

You can also use the provided startup script to launch the app. This script will:
- Activate the virtual environment (if present)
- Start the Streamlit app

From the project root directory, run:

```bash
bash startup.sh
```

This is the recommended way to start the app if you have a virtual environment set up in the project.

## Usage

### Configuration

- The app automatically reads your Pinecone and OpenAI API keys from the `.env` file in the project root
- You can also manually enter your API keys, index name, and namespace in the sidebar
- The default Pinecone index is `msft-deep-search-oai-3-small` (host: `https://msft-deep-search-oai-3-small-q0sfhsc.svc.aped-4627-b74a.pinecone.io`)

### Searching

1. Go to the "Search Similar Texts" tab
2. Enter your search query in the text area
3. Adjust the number of results using the slider
4. Click "Search" to find semantically similar texts in your Pinecone index
5. View results with similarity scores and metadata

### Inserting Text

1. Go to the "Insert Text" tab
2. Enter the text you want to store in Pinecone
3. Add optional metadata key-value pairs (e.g., author, category, date)
4. Click "Insert Text" to add your content to Pinecone
5. View the generated vector ID and confirmation details

## Technical Notes

- Uses OpenAI's `text-embedding-3-small` model for embedding all text and queries
- Pinecone is used for vector storage and fast semantic search
- The app and scripts are compatible with Pinecone v3+ and OpenAI's latest API
- The app connects to the backend modules by ensuring the project root is in the Python path 