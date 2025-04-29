"""
Validator module for Pinecone search and insert operations.
Provides functions to validate inputs before executing Pinecone operations.
"""

def validate_search_params(query_text, top_k, api_key, index_name):
    """
    Validate parameters for search_similar_texts function.
    
    Args:
        query_text (str): The text to search for similar items
        top_k (int): Number of similar results to return
        api_key (str): Pinecone API key
        index_name (str): Name of the Pinecone index to search
        
    Raises:
        ValueError: If any parameter is invalid
        TypeError: If parameter types don't match expectations
    """
    # Check for empty query
    if not query_text or not isinstance(query_text, str) or query_text.strip() == "":
        raise ValueError("Query text cannot be empty and must be a string")
    
    # Check top_k is a positive integer
    if not isinstance(top_k, int):
        raise TypeError("top_k must be an integer")
    if top_k <= 0:
        raise ValueError("top_k must be a positive integer")
    
    # Check API key
    if not api_key:
        raise ValueError("API key is required")
    
    # Check index name
    if not index_name:
        raise ValueError("Index name is required")

def validate_insert_params(text, metadata, api_key, index_name):
    """
    Validate parameters for insert_text function.
    
    Args:
        text (str): The text to insert
        metadata (dict): Additional metadata to store
        api_key (str): Pinecone API key
        index_name (str): Name of the Pinecone index
        
    Raises:
        ValueError: If any parameter is invalid
        TypeError: If parameter types don't match expectations
    """
    # Check for empty text
    if not text or not isinstance(text, str) or text.strip() == "":
        raise ValueError("Text cannot be empty and must be a string")
    
    # Check metadata is a dict if provided
    if metadata is not None and not isinstance(metadata, dict):
        raise TypeError("Metadata must be a dictionary")
    
    # Check API key
    if not api_key:
        raise ValueError("API key is required")
    
    # Check index name
    if not index_name:
        raise ValueError("Index name is required") 