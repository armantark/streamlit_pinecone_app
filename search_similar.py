# search_similar.py
import os
import openai
import pinecone
from dotenv import load_dotenv
from search_validator import validate_search_params

# Ensure .env is loaded
load_dotenv(dotenv_path=".env")

def embed_text_openai(text, openai_api_key):
    openai.api_key = openai_api_key
    response = openai.embeddings.create(
        input=[text],
        model="text-embedding-3-small"
    )
    return response.data[0].embedding

def search_similar_texts(query_text, top_k=5, api_key=None, index_name=None, namespace="", openai_api_key=None, pinecone_host=None):
    """
    Search for similar texts to the query using Pinecone vectordb and OpenAI embedding.
    Args:
        query_text (str): The text to search for similar items
        top_k (int): Number of similar results to return
        api_key (str): Pinecone API key (if not provided, will check env vars)
        index_name (str): Name of the Pinecone index to search
        namespace (str): Namespace to use in the index
        openai_api_key (str): OpenAI API key
        pinecone_host (str): Pinecone index host URL (optional)
    Returns:
        list: Top-k similar texts with their similarity scores
    """
    if index_name is None:
        index_name = os.getenv("PINECONE_INDEX_NAME", "msft-deep-search-oai-3-small")
    validate_search_params(query_text, top_k, api_key, index_name)
    load_dotenv()
    pinecone_api_key = api_key or os.getenv("PINECONE_API_KEY")
    openai_api_key = openai_api_key or os.getenv("OPENAI_API_KEY")
    pinecone_host = pinecone_host or os.getenv("PINECONE_HOST")
    if not pinecone_api_key:
        raise ValueError("Pinecone API key not provided and not found in environment variables")
    if not openai_api_key:
        raise ValueError("OpenAI API key not provided and not found in environment variables")
    pc = pinecone.Pinecone(api_key=pinecone_api_key)
    if pinecone_host:
        index = pc.Index(index_name, host=pinecone_host)
    else:
        index = pc.Index(index_name)
    try:
        stats = index.describe_index_stats()
        total_vectors = stats.get('total_vector_count', 0)
        if total_vectors == 0:
            return []
    except Exception as e:
        print(f"Error fetching index stats: {e}")
    # Embed the query using OpenAI
    query_embedding = embed_text_openai(query_text, openai_api_key)
    # Search using the dense vector
    results = index.query(
        namespace=namespace,
        vector=query_embedding,
        top_k=top_k,
        include_metadata=True
    )
    similar_items = []
    matches = results.matches or []
    for match in matches:
        item_id = match["id"]
        similarity_score = match["score"]
        metadata = match.get("metadata", {})
        text = metadata.get("text", "No text available")
        similar_items.append({
            "id": item_id,
            "text": text,
            "similarity_score": similarity_score,
            "metadata": metadata
        })
    return similar_items

def main():
    import argparse
    parser = argparse.ArgumentParser(description="Search for similar texts in Pinecone vectordb using OpenAI embedding")
    parser.add_argument("query", type=str, help="The query text to search for")
    parser.add_argument("--top-k", type=int, default=5, help="Number of similar results to return")
    parser.add_argument("--api-key", type=str, help="Pinecone API key (optional if in .env file)")
    parser.add_argument("--openai-api-key", type=str, help="OpenAI API key (optional if in .env file)")
    parser.add_argument("--index-name", type=str, default="msft-deep-search-oai-3-small", help="Pinecone index name")
    parser.add_argument("--namespace", type=str, default="", help="Namespace in the Pinecone index")
    parser.add_argument("--pinecone-host", type=str, help="Pinecone index host URL (optional)")
    args = parser.parse_args()
    try:
        results = search_similar_texts(
            args.query,
            top_k=args.top_k,
            api_key=args.api_key,
            index_name=args.index_name,
            namespace=args.namespace,
            openai_api_key=args.openai_api_key,
            pinecone_host=args.pinecone_host
        )
        print(f"Top {args.top_k} similar results for query: '{args.query}'\n")
        for i, item in enumerate(results, 1):
            print(f"{i}. ID: {item['id']}")
            print(f"   Text: {item['text'][:100]}..." if len(item['text']) > 100 else f"   Text: {item['text']}")
            print(f"   Similarity Score: {item['similarity_score']:.4f}")
            print()
    except Exception as e:
        print(f"Error: {str(e)}")

if __name__ == "__main__":
    main() 