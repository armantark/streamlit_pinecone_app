import sys
import os
import uuid
import time
from typing import List, Dict, Any, Optional

# Add parent directory to path to access the original modules
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from search_similar import search_similar_texts
from insert_text import insert_text
from dotenv import load_dotenv
from pinecone import Pinecone

# Load environment variables
load_dotenv(dotenv_path=os.path.join(os.path.dirname(__file__), '..', '.env'))

# Configuration
api_key = os.getenv("PINECONE_API_KEY")
# Use "your-test-index" as fallback, but this should be configured in .env
index_name = os.getenv("PINECONE_INDEX_NAME")

# If index name is still not available, try to detect available indexes
if not index_name:
    try:
        pc = Pinecone(api_key=api_key)
        indexes = pc.list_indexes()
        if indexes:
            # Use the first available index
            index_name = indexes[0].name
            print(f"Automatically selected index: {index_name}")
        else:
            print("No indexes found in your Pinecone account. Please specify an index name.")
            exit(1)
    except Exception as e:
        print(f"Error listing indexes: {e}")
        print("Please provide an index name in your .env file as PINECONE_INDEX_NAME")
        exit(1)

# Test data with a unique prefix to identify and clean up
TEST_PREFIX = f"test_{uuid.uuid4().hex[:8]}"
test_vectors = []

def print_separator(title: str) -> None:
    """Print a separator with a title for better readability."""
    print("\n" + "=" * 50)
    print(f" {title} ".center(50, "="))
    print("=" * 50 + "\n")

def create_test_data() -> List[str]:
    """
    Create test data in Pinecone index and return the generated IDs.
    """
    print_separator("CREATING TEST DATA")
    
    test_texts = [
        f"{TEST_PREFIX} This is a sample document about artificial intelligence and machine learning",
        f"{TEST_PREFIX} Python programming is widely used for data science and web development",
        f"{TEST_PREFIX} Vector databases like Pinecone help with semantic search capabilities",
        f"{TEST_PREFIX} Cloud computing services offer scalable solutions for businesses",
        f"{TEST_PREFIX} Natural language processing helps computers understand human text"
    ]
    
    vector_ids = []
    for i, text in enumerate(test_texts):
        try:
            # Add metadata to help with cleanup
            metadata = {
                "test_data": True,
                "test_prefix": TEST_PREFIX,
                "test_index": i
            }
            
            # Insert the test text
            vector_id = insert_text(
                text=text,
                metadata=metadata,
                api_key=api_key,
                index_name=index_name,
                namespace=""  # Use default namespace
            )
            
            vector_ids.append(vector_id)
            test_vectors.append(vector_id)  # Keep track for cleanup
            print(f"✅ Created test vector {i+1}/{len(test_texts)}: ID={vector_id}")
            
        except Exception as e:
            print(f"❌ Failed to create test vector {i+1}: {e}")
    
    # Wait for indexing to complete
    print("Waiting for indexing to complete...")
    time.sleep(2)
    
    return vector_ids

def test_search_functionality() -> None:
    """
    Test the search functionality with different queries.
    """
    print_separator("TESTING SEARCH FUNCTIONALITY")
    
    test_queries = [
        "artificial intelligence",
        "python programming",
        "vector databases",
        "cloud computing",
        "natural language processing"
    ]
    
    for i, query in enumerate(test_queries):
        try:
            print(f"\nQuery {i+1}: '{query}'")
            
            # Search for similar texts
            results = search_similar_texts(
                query_text=query,
                top_k=3,
                api_key=api_key,
                index_name=index_name
            )
            
            if results:
                print(f"✅ Found {len(results)} results")
                
                # Check if any of our test vectors are in the results
                test_hits = [r for r in results if r['id'] in test_vectors]
                
                if test_hits:
                    print(f"✅ {len(test_hits)}/{len(results)} results are from our test data")
                    
                    # Display the first test hit with similarity score
                    for j, hit in enumerate(test_hits[:2]):
                        print(f"  Result {j+1}:")
                        print(f"  - ID: {hit['id']}")
                        print(f"  - Score: {hit['similarity_score']:.4f}")
                        print(f"  - Text: {hit['text'][:100]}...")
                else:
                    print("❌ None of our test vectors were found in the results")
            else:
                print("❌ No results found")
                
        except Exception as e:
            print(f"❌ Search failed: {e}")

def cleanup_test_data() -> None:
    """
    Clean up the test data from the Pinecone index.
    """
    print_separator("CLEANING UP TEST DATA")
    
    if not test_vectors:
        print("No test vectors to clean up.")
        return
    
    try:
        # Connect to Pinecone
        pc = Pinecone(api_key=api_key)
        index = pc.Index(index_name)
        
        # Delete vectors by ID
        index.delete(ids=test_vectors)
        
        print(f"✅ Successfully deleted {len(test_vectors)} test vectors")
        
    except Exception as e:
        print(f"❌ Failed to clean up test data: {e}")
        print("Manual cleanup may be required with the following IDs:")
        for vid in test_vectors:
            print(f"  - {vid}")

def test_metadata_search() -> None:
    """
    Test that metadata is correctly stored and retrieved.
    """
    print_separator("TESTING METADATA FUNCTIONALITY")
    
    try:
        # Connect to Pinecone
        pc = Pinecone(api_key=api_key)
        index = pc.Index(index_name)
        
        # Fetch one of our test vectors to check metadata
        if test_vectors:
            vector_id = test_vectors[0]
            response = index.fetch(ids=[vector_id])
            
            if vector_id in response['vectors']:
                vector = response['vectors'][vector_id]
                metadata = vector.get('metadata', {})
                
                if metadata:
                    print(f"✅ Retrieved metadata for vector {vector_id}:")
                    for key, value in metadata.items():
                        print(f"  - {key}: {value}")
                    
                    # Verify test metadata fields are present
                    if metadata.get('test_data') and metadata.get('test_prefix') == TEST_PREFIX:
                        print("✅ Test metadata fields verified")
                    else:
                        print("❌ Test metadata fields are missing or incorrect")
                else:
                    print(f"❌ No metadata found for vector {vector_id}")
            else:
                print(f"❌ Vector {vector_id} not found in the index")
    except Exception as e:
        print(f"❌ Metadata test failed: {e}")

def run_tests():
    """
    Run all integration tests with cleanup.
    """
    print_separator("STARTING INTEGRATION TESTS")
    print(f"Using index: {index_name}")
    print(f"Test prefix: {TEST_PREFIX}")
    
    try:
        # Step 1: Create test data
        create_test_data()
        
        # Step 2: Test search functionality
        test_search_functionality()
        
        # Step 3: Test metadata functionality
        test_metadata_search()
        
    except Exception as e:
        print(f"❌ Tests failed with error: {e}")
    finally:
        # Always clean up, even if tests fail
        cleanup_test_data()
        
    print_separator("TESTS COMPLETED")

if __name__ == "__main__":
    run_tests() 