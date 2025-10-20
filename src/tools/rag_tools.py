"""
RAG (Retrieval-Augmented Generation) Tools
Similarity search using OpenSearch Vector Database
"""

from strands import tool
from opensearch_utils import OpenSearchVectorDB
import os
from dotenv import load_dotenv

load_dotenv()

# Initialize OpenSearch client
_opensearch_client = None


def _get_opensearch_client():
    """Lazy initialization of OpenSearch client"""
    global _opensearch_client
    if _opensearch_client is None:
        _opensearch_client = OpenSearchVectorDB(
            host=os.getenv('RAG_OPENSEARCH_HOST'),
            username=os.getenv('RAG_OPENSEARCH_USERNAME'),
            password=os.getenv('RAG_OPENSEARCH_PASSWORD'),
            port=int(os.getenv('RAG_OPENSEARCH_PORT', '443'))
        )
    return _opensearch_client


@tool
def similarity_search(query: str, user_id: str = "default_user", top_k: int = 3) -> str:
    """
    Perform semantic similarity search on user's knowledge base using vector embeddings.
    
    This tool searches through previously stored documents and returns the most relevant
    information based on semantic similarity to the query.
    
    Args:
        query: The search query or question to find relevant information for
        user_id: User identifier for accessing their personal knowledge base (default: "default_user")
        top_k: Number of most relevant results to return (default: 3, max: 10)
    
    Returns:
        Formatted string containing the most relevant documents with their similarity scores
        and metadata. Returns error message if search fails.
    
    Example:
        similarity_search("What is the difference between CPU and GPU?", user_id="user123", top_k=3)
    """
    try:
        client = _get_opensearch_client()
        
        # Perform similarity search
        results = client.similarity_search(
            user_id=user_id,
            query_text=query,
            k=min(top_k, 10)  # Limit to max 10 results
        )
        
        if not results:
            return f"No relevant information found for query: '{query}'"
        
        # Format results
        formatted_results = [f"Found {len(results)} relevant documents:\n"]
        
        for i, result in enumerate(results, 1):
            formatted_results.append(f"\n--- Result {i} (Relevance Score: {result['score']:.4f}) ---")
            formatted_results.append(f"Content: {result['text']}")
            
            if result.get('metadata'):
                metadata_str = ", ".join([f"{k}: {v}" for k, v in result['metadata'].items()])
                formatted_results.append(f"Metadata: {metadata_str}")
            
            if result.get('timestamp'):
                formatted_results.append(f"Indexed: {result['timestamp']}")
        
        return "\n".join(formatted_results)
        
    except ValueError as e:
        return f"Error: {str(e)}. The knowledge base for user '{user_id}' may not exist yet."
    except Exception as e:
        return f"Error performing similarity search: {str(e)}"
