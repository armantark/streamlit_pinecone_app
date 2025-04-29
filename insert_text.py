# insert_text.py
import os
import argparse
import uuid
import openai
from pinecone import Pinecone
from dotenv import load_dotenv
from search_validator import validate_insert_params

# Ensure .env is loaded
load_dotenv(dotenv_path=".env")

def embed_text_openai(text, openai_api_key):
    openai.api_key = openai_api_key
    response = openai.embeddings.create(
        input=[text],
        model="text-embedding-3-small"
    )
    return response.data[0].embedding

def insert_text(text, metadata=None, api_key=None, index_name=None, namespace="", openai_api_key=None, pinecone_host=None):
    """
    Insert text into Pinecone vectordb using OpenAI embedding.
    Args:
        text (str): The text to insert
        metadata (dict): Additional metadata to store with the vector
        api_key (str): Pinecone API key (if not provided, will check env vars)
        index_name (str): Name of the Pinecone index
        namespace (str): Namespace to use in the index
        openai_api_key (str): OpenAI API key
        pinecone_host (str): Pinecone index host URL (optional)
    Returns:
        str: ID of the inserted vector
    """
    if index_name is None:
        index_name = os.getenv("PINECONE_INDEX_NAME", "msft-deep-search-oai-3-small")
    validate_insert_params(text, metadata, api_key, index_name)
    pinecone_api_key = api_key or os.getenv("PINECONE_API_KEY")
    openai_api_key = openai_api_key or os.getenv("OPENAI_API_KEY")
    pinecone_host = pinecone_host or os.getenv("PINECONE_HOST")
    if not pinecone_api_key:
        raise ValueError("Pinecone API key not provided and not found in environment variables")
    if not openai_api_key:
        raise ValueError("OpenAI API key not provided and not found in environment variables")
    pc = Pinecone(api_key=pinecone_api_key)
    if pinecone_host:
        index = pc.Index(index_name, host=pinecone_host)
    else:
        index = pc.Index(index_name)
    vector_id = str(uuid.uuid4())
    embedding = embed_text_openai(text, openai_api_key)
    if metadata is None:
        metadata = {}
    metadata["text"] = text
    index.upsert(
        vectors=[{
            "id": vector_id,
            "values": embedding,
            "metadata": metadata
        }],
        namespace=namespace
    )
    return vector_id

def main():
    parser = argparse.ArgumentParser(description="Insert text into Pinecone vectordb using OpenAI embedding")
    parser.add_argument("text", type=str, help="The text to insert")
    parser.add_argument("--metadata-key", action="append", help="Additional metadata key (can be used multiple times)")
    parser.add_argument("--metadata-value", action="append", help="Additional metadata value (can be used multiple times)")
    parser.add_argument("--api-key", type=str, help="Pinecone API key (optional if in .env file)")
    parser.add_argument("--openai-api-key", type=str, help="OpenAI API key (optional if in .env file)")
    parser.add_argument("--index-name", type=str, default="msft-deep-search-oai-3-small", help="Pinecone index name")
    parser.add_argument("--namespace", type=str, default="", help="Namespace in the Pinecone index")
    parser.add_argument("--pinecone-host", type=str, help="Pinecone index host URL (optional)")
    args = parser.parse_args()
    metadata = {}
    if args.metadata_key and args.metadata_value:
        if len(args.metadata_key) != len(args.metadata_value):
            print("Error: Number of metadata keys and values must match")
            return
        for key, value in zip(args.metadata_key, args.metadata_value):
            metadata[key] = value
    try:
        vector_id = insert_text(
            args.text,
            metadata=metadata,
            api_key=args.api_key,
            index_name=args.index_name,
            namespace=args.namespace,
            openai_api_key=args.openai_api_key,
            pinecone_host=args.pinecone_host
        )
        print(f"Successfully inserted text into Pinecone")
        print(f"Vector ID: {vector_id}")
        print(f"Index: {args.index_name}")
        print(f"Namespace: {args.namespace or 'default'}")
    except Exception as e:
        print(f"Error: {str(e)}")

if __name__ == "__main__":
    main() 