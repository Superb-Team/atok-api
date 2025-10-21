# ATOK AI Notetaker API

API for audio transcription, text enhancement, and vector search powered by OpenSearch.

## Features

1. **Audio Transcription** – Transcribe audio files using Amazon Transcribe.
2. **Text Enhancement** – Improve transcribed text using AI models.
3. **Combined Transcribe+Enhance** – One-call transcription and enhancement.
4. **Multi-Agent System** – Task management, knowledge retrieval, and **MCP Servers** operations.
5. **Vector Search** – Store and search documents in OpenSearch using Cohere Embed v4.

## Quick Start

### Installation

```bash
# Install dependencies
pip install -r requirements.txt

# Or using uv
uv pip install -r requirements.txt
```

### Environment Variables

Create a `.env` file:

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

Server runs at `http://localhost:8000`

## API Endpoints

### Authentication

All endpoints require an API key in the header:

```
X-API-Key: your-api-key-here
```

### 1. Transcription

**POST** `/transcribe`
Upload an audio file for transcription.

```bash
curl -X POST "http://localhost:8000/transcribe" \
  -H "X-API-Key: your-api-key" \
  -F "file=@audio.mp3"
```

### 2. Text Enhancement

**POST** `/enhance`
Enhance transcribed text using AI.

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
One endpoint for both transcription and enhancement.

```bash
curl -X POST "http://localhost:8000/transcribe-enhance" \
  -H "X-API-Key: your-api-key" \
  -F "file=@audio.mp3" \
  -F "context=meeting notes"
```

### 4. Multi-Agent System

**POST** `/agent/invoke` – Interact with the multi-agent system.
**POST** `/agent/stream` – Stream real-time agent responses (SSE).
**GET** `/agent/agents` – List all available agents.

```bash
# Task management
curl -X POST "http://localhost:8000/agent/invoke" \
  -H "X-API-Key: your-api-key" \
  -H "Content-Type: application/json" \
  -d '{"prompt": "Create a task to review code", "user_id": "user123"}'

# Knowledge retrieval
curl -X POST "http://localhost:8000/agent/invoke" \
  -H "X-API-Key: your-api-key" \
  -H "Content-Type: application/json" \
  -d '{"prompt": "Search for authentication info", "user_id": "user123"}'
```


### 5. OpenSearch Vector Search

**Note:** Every OpenSearch operation requires a `user_id`.

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

## API Documentation

### Interactive Docs

* **Swagger UI:** `http://localhost:8000/docs`
* **ReDoc:** `http://localhost:8000/redoc`


## Project Structure

```
.
├── src/
│   ├── main.py                   # FastAPI main entry
│   ├── opensearch_client.py      # OpenSearch client & operations
│   ├── opensearch_endpoints.py   # OpenSearch endpoints
│   ├── enhance/                  # Text enhancement module
│   ├── transcribe/               # Transcription module
│   ├── vector/                   # Vector operations
│   └── agents/                   # MCP Agent system
├── database/                     # Database schema & queries
├── .env                          # Environment variables
├── requirements.txt              # Dependencies
├── README.md                     # This file
└── OPENSEARCH_API_GUIDE.md       # OpenSearch API guide
```

## Development

### Docker

```bash
docker build -t atok-api .
docker run -p 8000:8000 --env-file .env atok-api
```

## Notes

* `user_id` is **required** for all OpenSearch operations.
* Each user has an isolated collection for privacy.
* Default vector dimension: **1536** (Cohere Embed v4).
* Cohere model is available in **ap-northeast-1** region.

## License

MIT
