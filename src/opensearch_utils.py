import boto3
import json
from opensearchpy import OpenSearch, RequestsHttpConnection
from typing import List, Dict, Any, Optional
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

class OpenSearchVectorDB:
    """
    AWS OpenSearch Service client dengan Cohere Embed v4.
    Setiap user memiliki collection (index) sendiri untuk privacy.
    """
    
    def __init__(
        self,
        host: str,
        username: str,
        password: str,
        port: int = 443,
        use_ssl: bool = True
    ):
        """Initialize OpenSearch client dengan basic auth."""
        self.host = host
        self.client = OpenSearch(
            hosts=[{'host': host, 'port': port}],
            http_auth=(username, password),
            use_ssl=use_ssl,
            verify_certs=True,
            connection_class=RequestsHttpConnection,
            timeout=30
        )
        
        # Initialize Bedrock client - Cohere v4 available di ap-northeast-1
        session = boto3.Session(
            aws_access_key_id=os.getenv('RAG_AWS_ACCESS_KEY_ID'),
            aws_secret_access_key=os.getenv('RAG_AWS_SECRET_ACCESS_KEY'),
            region_name=os.getenv('RAG_AWS_REGION', 'ap-northeast-1')
        )
        self.bedrock_runtime = session.client('bedrock-runtime')
    
    def create_user_collection(
        self,
        user_id: str,
        vector_dimension: int = 1536,
        index_settings: Optional[Dict] = None
    ) -> Dict[str, Any]:
        """Membuat collection (index) untuk user tertentu."""
        index_name = f"user_{user_id}_collection"
        
        if index_settings is None:
            index_settings = {
                "settings": {
                    "index": {
                        "knn": True,
                        "knn.algo_param.ef_search": 512,
                        "number_of_shards": 1,
                        "number_of_replicas": 1
                    }
                },
                "mappings": {
                    "properties": {
                        "text": {"type": "text"},
                        "vector": {
                            "type": "knn_vector",
                            "dimension": vector_dimension,
                            "method": {
                                "name": "hnsw",
                                "space_type": "cosinesimil",
                                "engine": "faiss",  # Changed from nmslib to faiss
                                "parameters": {
                                    "ef_construction": 512,
                                    "m": 16
                                }
                            }
                        },
                        "metadata": {"type": "object", "enabled": True},
                        "timestamp": {"type": "date"}
                    }
                }
            }
        
        if self.client.indices.exists(index=index_name):
            return {
                "status": "exists",
                "message": f"Collection {index_name} sudah ada",
                "index_name": index_name
            }
        
        response = self.client.indices.create(index=index_name, body=index_settings)
        
        return {
            "status": "created",
            "message": f"Collection {index_name} berhasil dibuat",
            "index_name": index_name,
            "response": response
        }
    
    def text_to_vector(self, text: str, model_id: str = None) -> List[float]:
        """Convert text ke vector menggunakan Cohere Embed v4."""
        if model_id is None:
            model_id = os.getenv('RAG_COHERE_MODEL_ID', 'cohere.embed-v4:0')
        
        body = json.dumps({
            "texts": [text],
            "input_type": "search_document",
            "embedding_types": ["float"],
            "truncate": "RIGHT"
        })
        
        response = self.bedrock_runtime.invoke_model(
            modelId=model_id,
            body=body,
            contentType="application/json",
            accept="*/*"
        )
        
        response_body = json.loads(response['body'].read())
        embeddings = response_body.get('embeddings', {})
        float_embeddings = embeddings.get('float', [[]])
        
        if not float_embeddings or len(float_embeddings) == 0:
            raise ValueError("No embeddings returned from model")
        
        return float_embeddings[0]
    
    def query_to_vector(self, text: str, model_id: str = None) -> List[float]:
        """Convert query text ke vector (optimized untuk search)."""
        if model_id is None:
            model_id = os.getenv('RAG_COHERE_MODEL_ID', 'cohere.embed-v4:0')
        
        body = json.dumps({
            "texts": [text],
            "input_type": "search_query",
            "embedding_types": ["float"],
            "truncate": "RIGHT"
        })
        
        response = self.bedrock_runtime.invoke_model(
            modelId=model_id,
            body=body,
            contentType="application/json",
            accept="*/*"
        )
        
        response_body = json.loads(response['body'].read())
        embeddings = response_body.get('embeddings', {})
        float_embeddings = embeddings.get('float', [[]])
        
        if not float_embeddings or len(float_embeddings) == 0:
            raise ValueError("No embeddings returned from model")
        
        return float_embeddings[0]
    
    def insert_document(
        self,
        user_id: str,
        text: str,
        metadata: Optional[Dict] = None,
        doc_id: Optional[str] = None
    ) -> Dict[str, Any]:
        """Insert document (text + vector) ke user collection."""
        index_name = f"user_{user_id}_collection"
        
        if not self.client.indices.exists(index=index_name):
            raise ValueError(f"Collection {index_name} tidak ditemukan")
        
        vector = self.text_to_vector(text)
        
        from datetime import datetime
        document = {
            "text": text,
            "vector": vector,
            "metadata": metadata or {},
            "timestamp": datetime.utcnow().isoformat()
        }
        
        if doc_id:
            response = self.client.index(index=index_name, id=doc_id, body=document)
        else:
            response = self.client.index(index=index_name, body=document)
        
        return {
            "status": "success",
            "index_name": index_name,
            "doc_id": response['_id'],
            "response": response
        }
    
    def bulk_insert_documents(
        self,
        user_id: str,
        documents: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Bulk insert multiple documents."""
        index_name = f"user_{user_id}_collection"
        
        if not self.client.indices.exists(index=index_name):
            raise ValueError(f"Collection {index_name} tidak ditemukan")
        
        from datetime import datetime
        bulk_body = []
        
        for doc in documents:
            text = doc.get('text')
            if not text:
                continue
            
            vector = self.text_to_vector(text)
            
            action = {"index": {"_index": index_name}}
            if 'doc_id' in doc:
                action["index"]["_id"] = doc['doc_id']
            
            document = {
                "text": text,
                "vector": vector,
                "metadata": doc.get('metadata', {}),
                "timestamp": datetime.utcnow().isoformat()
            }
            
            bulk_body.append(action)
            bulk_body.append(document)
        
        response = self.client.bulk(body=bulk_body)
        
        items = response.get('items', [])
        success_count = sum(1 for item in items if item.get('index', {}).get('status') in [200, 201])
        error_count = len(items) - success_count
        
        return {
            "status": "completed",
            "total": len(items),
            "success": success_count,
            "errors": error_count,
            "response": response
        }
    
    def similarity_search(
        self,
        user_id: str,
        query_text: str,
        k: int = 5,
        min_score: Optional[float] = None,
        filter_metadata: Optional[Dict] = None
    ) -> List[Dict[str, Any]]:
        """Similarity search menggunakan vector similarity."""
        index_name = f"user_{user_id}_collection"
        
        if not self.client.indices.exists(index=index_name):
            raise ValueError(f"Collection {index_name} tidak ditemukan")
        
        query_vector = self.query_to_vector(query_text)
        
        knn_query = {
            "size": k,
            "query": {
                "knn": {
                    "vector": {
                        "vector": query_vector,
                        "k": k
                    }
                }
            },
            "_source": ["text", "metadata", "timestamp"]
        }
        
        if filter_metadata:
            knn_query["query"] = {
                "bool": {
                    "must": [
                        {"knn": {"vector": {"vector": query_vector, "k": k}}}
                    ],
                    "filter": [
                        {"term": {f"metadata.{key}": value}}
                        for key, value in filter_metadata.items()
                    ]
                }
            }
        
        response = self.client.search(index=index_name, body=knn_query)
        
        results = []
        for hit in response['hits']['hits']:
            score = hit['_score']
            
            if min_score and score < min_score:
                continue
            
            results.append({
                "doc_id": hit['_id'],
                "score": score,
                "text": hit['_source'].get('text'),
                "metadata": hit['_source'].get('metadata', {}),
                "timestamp": hit['_source'].get('timestamp')
            })
        
        return results
    
    def delete_user_collection(self, user_id: str) -> Dict[str, Any]:
        """Hapus collection user."""
        index_name = f"user_{user_id}_collection"
        
        if not self.client.indices.exists(index=index_name):
            return {
                "status": "not_found",
                "message": f"Collection {index_name} tidak ditemukan"
            }
        
        response = self.client.indices.delete(index=index_name)
        
        return {
            "status": "deleted",
            "message": f"Collection {index_name} berhasil dihapus",
            "response": response
        }
    
    def get_collection_stats(self, user_id: str) -> Dict[str, Any]:
        """Get statistics dari user collection."""
        index_name = f"user_{user_id}_collection"
        
        if not self.client.indices.exists(index=index_name):
            raise ValueError(f"Collection {index_name} tidak ditemukan")
        
        stats = self.client.indices.stats(index=index_name)
        count = self.client.count(index=index_name)
        
        return {
            "index_name": index_name,
            "document_count": count['count'],
            "size_in_bytes": stats['indices'][index_name]['total']['store']['size_in_bytes'],
            "stats": stats
        }
