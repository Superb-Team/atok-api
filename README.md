# ATOK AI Notetaker API

API untuk audio transcription, text enhancement, dan vector search menggunakan OpenSearch.

## Features

1. **Audio Transcription** - Transcribe audio files menggunakan Amazon Transcribe
2. **Text Enhancement** - Enhance transcribed text menggunakan AI
3. **Combined Transcribe+Enhance** - One-call transcription and enhancement
4. **Multi-Agent System** - Task management, knowledge search, GitHub operations
5. **Vector Search** - Store dan search documents menggunakan OpenSearch dengan Cohere Embed v4

## Quick Start

### Installation

```bash
# Install dependencies
pip install -r requirements.txt

# Or using uv
uv pip install -r requirements.txt
```

### Environment Variables

Create `.env` file:

```env
# API Key
API_KEY=your-api-key-here

# AWS Transcribe & S3
AWS_ACCESS_KEY_ID=your-aws-key
AWS_SECRET_ACCESS_KEY=your-aws-secret
AWS_REGION=ap-southeast-1
S3_BUCKET_NAME=your-bucket-name

# OpenSearch
OPENSEARCH_HOST=your-opensearch-host.com
OPENSEARCH_USERNAME=admin
OPENSEARCH_PASSWORD=your-password
OPENSEARCH_PORT=443

# AWS Bedrock for Cohere Embed v4
RAG_AWS_ACCESS_KEY_ID=your-aws-key
RAG_AWS_SECRET_ACCESS_KEY=your-aws-secret
RAG_AWS_REGION=ap-northeast-1
RAG_COHERE_MODEL_ID=cohere.embed-v4:0
```

### Run Server

```bash
# Development
python src/main.py

# Or using uvicorn
uvicorn src.main:app --reload --host 0.0.0.0 --port 8000
```

Server akan berjalan di `http://localhost:8000`

## API Endpoints

### Authentication
Semua endpoint memerlukan API Key di header:
```
X-API-Key: your-api-key-here
```

### 1. Transcription

**POST** `/transcribe`

Upload audio file untuk transcription.

```bash
curl -X POST "http://localhost:8000/transcribe" \
  -H "X-API-Key: your-api-key" \
  -F "file=@audio.mp3"
```

### 2. Text Enhancement

**POST** `/enhance`

Enhance transcribed text menggunakan AI.

```bash
curl -X POST "http://localhost:8000/enhance" \
  -H "X-API-Key: your-api-key" \
  -H "Content-Type: application/json" \
  -d '{
    "text": "transcribed text here",
    "context": "optional context"
  }'
```

### 3. Transcribe + Enhance (Combined)

**POST** `/transcribe-enhance`

Combined endpoint: Transcribe dan enhance audio dalam satu call.

```bash
curl -X POST "http://localhost:8000/transcribe-enhance" \
  -H "X-API-Key: your-api-key" \
  -F "file=@audio.mp3" \
  -F "context=meeting notes"
```

**Lihat dokumentasi lengkap di [TEST_TRANSCRIBE_ENHANCE.md](./TEST_TRANSCRIBE_ENHANCE.md)**

### 4. Multi-Agent System

**POST** `/agent/invoke` - Interact dengan multi-agent system

**POST** `/agent/stream` - Streaming agent responses (SSE)

**GET** `/agent/agents` - List available agents

```bash
# Task management
curl -X POST "http://localhost:8000/agent/invoke" \
  -H "X-API-Key: your-api-key" \
  -H "Content-Type: application/json" \
  -d '{"prompt": "Create a task to review code", "user_id": "user123"}'

# Knowledge search
curl -X POST "http://localhost:8000/agent/invoke" \
  -H "X-API-Key: your-api-key" \
  -H "Content-Type: application/json" \
  -d '{"prompt": "Search for authentication info", "user_id": "user123"}'
```

**Lihat dokumentasi lengkap di [AGENT_API_GUIDE.md](./AGENT_API_GUIDE.md)**

### 5. OpenSearch Vector Search

**PENTING**: Semua endpoint OpenSearch memerlukan `user_id` sebagai input.

#### Check Collection
```bash
GET /opensearch/collection/check/{user_id}
```

#### Create Collection
```bash
POST /opensearch/collection/create
Body: {"user_id": "user123", "vector_dimension": 1536}
```

#### Insert Document
```bash
POST /opensearch/document/insert
Body: {
  "user_id": "user123",
  "text": "document content",
  "metadata": {"type": "note"}
}
```

#### Search Documents
```bash
POST /opensearch/search
Body: {
  "user_id": "user123",
  "query_text": "search query",
  "k": 5
}
```

**Lihat dokumentasi lengkap di [OPENSEARCH_API_GUIDE.md](./OPENSEARCH_API_GUIDE.md)**

## API Documentation

### Interactive Docs
- **Swagger UI**: `http://localhost:8000/docs`
- **ReDoc**: `http://localhost:8000/redoc`

### Complete Usage Guide
- **[USAGE.md](./USAGE.md)** - Complete guide untuk semua endpoint dengan contoh Postman
- **[ATOK_AI_API.postman_collection.json](./ATOK_AI_API.postman_collection.json)** - Import ke Postman untuk testing

## Project Structure

```
.
├── src/
│   ├── main.py                    # Main FastAPI application
│   ├── opensearch_client.py       # OpenSearch client & operations
│   ├── opensearch_endpoints.py    # OpenSearch endpoints (standalone)
│   ├── enhance/
│   │   └── enhance.py            # Text enhancement module
│   ├── transcribe/               # Transcription module
│   └── vector/                   # Vector operations
├── database/                     # Database schemas & queries
├── .env                         # Environment variables
├── requirements.txt             # Python dependencies
├── README.md                    # This file
└── OPENSEARCH_API_GUIDE.md     # OpenSearch API guide
```

## Development

### Testing OpenSearch

```bash
# Test simple connection
python test_opensearch_simple.py

# Test collection operations
python test_opensearch_collection.py
```

### Docker

```bash
# Build image
docker build -t atok-api .

# Run container
docker run -p 8000:8000 --env-file .env atok-api
```

## Notes

- **user_id** adalah WAJIB untuk semua operasi OpenSearch
- Setiap user memiliki collection terpisah untuk privacy
- Vector dimension default: 1536 (Cohere Embed v4)
- Cohere model tersedia di region `ap-northeast-1`

## License

MIT
