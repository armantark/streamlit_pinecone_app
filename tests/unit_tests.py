import sys
import os
import unittest
from unittest.mock import patch, MagicMock

# Add parent directory to path to access the original modules
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from search_similar import search_similar_texts
from insert_text import insert_text
from pinecone_app.search_validator import validate_search_params, validate_insert_params

class TestValidators(unittest.TestCase):
    """Unit tests for parameter validators."""
    
    def setUp(self):
        """Set up test environment."""
        self.api_key = "test_api_key"
        self.index_name = "test_index"
        self.test_text = "This is a test document for unit testing"
        self.test_query = "test document"
    
    def test_search_validator_empty_query(self):
        """Test that validator catches empty query."""
        with self.assertRaises(ValueError):
            validate_search_params("", 5, self.api_key, self.index_name)
    
    def test_search_validator_invalid_top_k(self):
        """Test that validator catches invalid top_k."""
        with self.assertRaises(TypeError):
            validate_search_params(self.test_query, "invalid", self.api_key, self.index_name)
        
        with self.assertRaises(ValueError):
            validate_search_params(self.test_query, 0, self.api_key, self.index_name)
            
        with self.assertRaises(ValueError):
            validate_search_params(self.test_query, -1, self.api_key, self.index_name)
    
    def test_search_validator_missing_api_key(self):
        """Test that validator catches missing API key."""
        with self.assertRaises(ValueError):
            validate_search_params(self.test_query, 5, None, self.index_name)
            
        with self.assertRaises(ValueError):
            validate_search_params(self.test_query, 5, "", self.index_name)
    
    def test_search_validator_missing_index(self):
        """Test that validator catches missing index name."""
        with self.assertRaises(ValueError):
            validate_search_params(self.test_query, 5, self.api_key, "")
            
        with self.assertRaises(ValueError):
            validate_search_params(self.test_query, 5, self.api_key, None)
    
    def test_insert_validator_empty_text(self):
        """Test that validator catches empty text."""
        with self.assertRaises(ValueError):
            validate_insert_params("", None, self.api_key, self.index_name)
            
        with self.assertRaises(ValueError):
            validate_insert_params("   ", None, self.api_key, self.index_name)
    
    def test_insert_validator_invalid_metadata(self):
        """Test that validator catches invalid metadata."""
        with self.assertRaises(TypeError):
            validate_insert_params(self.test_text, "invalid", self.api_key, self.index_name)
            
        with self.assertRaises(TypeError):
            validate_insert_params(self.test_text, 123, self.api_key, self.index_name)
    
    def test_insert_validator_missing_api_key(self):
        """Test that validator catches missing API key."""
        with self.assertRaises(ValueError):
            validate_insert_params(self.test_text, None, None, self.index_name)

class TestPineconeFunctions(unittest.TestCase):
    """Unit tests for the Pinecone app functions with mocked validators."""
    
    def setUp(self):
        """Set up test environment."""
        self.api_key = "test_api_key"
        self.index_name = "test_index"
        self.test_text = "This is a test document for unit testing"
        self.test_query = "test document"
    
    @patch('search_similar.validate_search_params')
    @patch('search_similar.Pinecone')
    def test_search_calls_validator(self, mock_pinecone, mock_validator):
        """Test that search_similar_texts calls validator."""
        # Mock the Pinecone client and Index
        mock_index = MagicMock()
        mock_pinecone.return_value.Index.return_value = mock_index
        
        # Mock the embeddings service
        mock_pinecone.return_value.inference.embed.return_value = [{'values': [0.1, 0.2, 0.3]}]
        
        # Mock query results
        mock_index.query.return_value = {
            "matches": [
                {
                    "id": "test_id_1",
                    "score": 0.95,
                    "metadata": {"text": "Test document one"}
                }
            ]
        }
        
        # Test search function
        search_similar_texts(
            query_text=self.test_query,
            top_k=2,
            api_key=self.api_key,
            index_name=self.index_name
        )
        
        # Verify validator was called with correct parameters
        mock_validator.assert_called_once_with(self.test_query, 2, self.api_key, self.index_name)
    
    @patch('insert_text.validate_insert_params')
    @patch('insert_text.Pinecone')
    def test_insert_calls_validator(self, mock_pinecone, mock_validator):
        """Test that insert_text calls validator."""
        # Mock the Pinecone client and Index
        mock_index = MagicMock()
        mock_pinecone.return_value.Index.return_value = mock_index
        
        # Mock the embeddings service
        mock_pinecone.return_value.inference.embed.return_value = [{'values': [0.1, 0.2, 0.3]}]
        
        # Test with metadata
        test_metadata = {"category": "test"}
        
        # Test insert function
        with patch('uuid.uuid4', return_value=MagicMock(hex='mock_uuid')):
            insert_text(
                text=self.test_text,
                metadata=test_metadata,
                api_key=self.api_key,
                index_name=self.index_name
            )
            
            # Verify validator was called with correct parameters
            mock_validator.assert_called_once_with(self.test_text, test_metadata, self.api_key, self.index_name)

if __name__ == '__main__':
    unittest.main() 