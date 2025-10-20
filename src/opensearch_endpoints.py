from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Optional, Dict, Any
import os
from dotenv import load_dotenv
from opensearch_client import OpenSearchVectorDB

load_dotenv()

app = FastAPI(title="OpenSearch Collection API")

# Initialize OpenSearch client
opensearch_client = OpenSearchVectorDB(
    host=os.getenv('OPENSEARCH_HOST'),
    username=os.getenv('OPENSEARCH_USERNAME'),
    password=os.getenv('OPENSEARCH_PASSWORD'),
    port=int(os.getenv('OPENSEARCH_PORT', 443)),
    use_ssl=True
)

class CollectionCheckResponse(BaseModel):
    exists: bool
    index_name: str
    message: str
    stats: Optional[Dict[str, Any]] = None

class CollectionCreateRequest(BaseModel):
    user_id: str
    vector_dimension: int = 1536

class CollectionCreateResponse(BaseModel):
    status: str
    message: str
    index_name: str
    response: Optional[Dict[str, Any]] = None

@app.get("/")
def root():
    """Health check endpoint"""
    return {
        "status": "ok",
        "service": "OpenSearch Collection API",
        "endpoints": {
            "check": "/collection/check/{user_id}",
            "create": "/collection/create",
            "stats": "/collection/stats/{user_id}",
            "delete": "/collection/delete/{user_id}"
        }
    }

@app.get("/collection/check/{user_id}", response_model=CollectionCheckResponse)
def check_collection(user_id: str):
    """
    Check apakah collection untuk user sudah ada
    
    Args:
        user_id: ID user (contoh: user69, user_211123)
    
    Returns:
        exists: True/False
        index_name: nama index
        message: status message
        stats: statistics jika collection ada
    """
    try:
        index_name = f"user_{user_id}_collection"
        
        # Check if index exists
        exists = opensearch_client.client.indices.exists(index=index_name)
        
        if exists:
            # Get stats if exists
            try:
                stats = opensearch_client.get_collection_stats(user_id)
                return CollectionCheckResponse(
                    exists=True,
                    index_name=index_name,
                    message=f"Collection {index_name} sudah ada",
                    stats=stats
                )
            except Exception as e:
                return CollectionCheckResponse(
                    exists=True,
                    index_name=index_name,
                    message=f"Collection {index_name} ada tapi gagal get stats: {str(e)}"
                )
        else:
            return CollectionCheckResponse(
                exists=False,
                index_name=index_name,
                message=f"Collection {index_name} belum ada"
            )
            
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error checking collection: {str(e)}")

@app.post("/collection/create", response_model=CollectionCreateResponse)
def create_collection(request: CollectionCreateRequest):
    """
    Buat collection baru untuk user
    
    Args:
        user_id: ID user
        vector_dimension: dimensi vector (default: 1536 untuk Cohere v4)
    
    Returns:
        status: created/exists
        message: status message
        index_name: nama index
    """
    try:
        result = opensearch_client.create_user_collection(
            user_id=request.user_id,
            vector_dimension=request.vector_dimension
        )
        
        return CollectionCreateResponse(
            status=result['status'],
            message=result['message'],
            index_name=result['index_name'],
            response=result.get('response')
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error creating collection: {str(e)}")

@app.delete("/collection/delete/{user_id}")
def delete_collection(user_id: str):
    """
    Hapus collection user (untuk testing/cleanup)
    
    Args:
        user_id: ID user
    """
    try:
        result = opensearch_client.delete_user_collection(user_id)
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error deleting collection: {str(e)}")

@app.get("/collection/stats/{user_id}")
def get_collection_stats(user_id: str):
    """
    Get statistics dari collection user
    
    Args:
        user_id: ID user
    """
    try:
        stats = opensearch_client.get_collection_stats(user_id)
        return stats
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error getting stats: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
