from fastapi import FastAPI, File, UploadFile, HTTPException, Form, Header, Depends
from fastapi.responses import JSONResponse, StreamingResponse
from fastapi.security import APIKeyHeader
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
import os
import tempfile
import json
from typing import Optional, Dict, Any, List
from dotenv import load_dotenv

# Import our modules
from enhance.enhance import llm, system_prompt
from langchain_core.messages import HumanMessage
from opensearch_client import OpenSearchVectorDB
from agents.supervisor import supervisor
from database import get_github_token

# Load environment variables
load_dotenv()

app = FastAPI(
    title="ATOK AI Notetaker API Endpoint",
    description="API for audio transcription, text enhancement, vector search, and multi-agent system",
    version="0.2.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# API Key Configuration
API_KEY = os.getenv("API_KEY")
api_key_header = APIKeyHeader(name="X-API-Key", auto_error=False)

async def verify_api_key(api_key: str = Depends(api_key_header)):
    """Verify API key from header"""
    if api_key != API_KEY:
        raise HTTPException(
            status_code=403,
            detail="Invalid or missing API Key"
        )
    return api_key

# Initialize OpenSearch client (lazy initialization)
_opensearch_client = None

def get_opensearch_client():
    """Get or create OpenSearch client"""
    global _opensearch_client
    if _opensearch_client is None:
        host = os.getenv('OPENSEARCH_HOST')
        if not host:
            raise ValueError("OPENSEARCH_HOST environment variable not set")
        _opensearch_client = OpenSearchVectorDB(
            host=host,
            username=os.getenv('OPENSEARCH_USERNAME'),
            password=os.getenv('OPENSEARCH_PASSWORD'),
            port=int(os.getenv('OPENSEARCH_PORT', 443)),
            use_ssl=True
        )
    return _opensearch_client

# Pydantic models
class TranscribeResponse(BaseModel):
    success: bool
    transcript: Optional[str] = None
    detected_languages: Optional[list] = None
    error: Optional[str] = None

class EnhanceRequest(BaseModel):
    text: str
    context: Optional[str] = None

class EnhanceResponse(BaseModel):
    success: bool
    original_text: str
    enhanced_text: Optional[str] = None
    error: Optional[str] = None

class TranscribeEnhanceResponse(BaseModel):
    success: bool
    transcript: Optional[str] = None
    enhanced_text: Optional[str] = None
    detected_languages: Optional[list] = None
    error: Optional[str] = None

# OpenSearch Models
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

class DocumentInsertRequest(BaseModel):
    user_id: str
    text: str
    metadata: Optional[Dict[str, Any]] = None
    doc_id: Optional[str] = None

class DocumentInsertResponse(BaseModel):
    status: str
    index_name: str
    doc_id: str
    response: Optional[Dict[str, Any]] = None

class BulkInsertRequest(BaseModel):
    user_id: str
    documents: List[Dict[str, Any]]

class BulkInsertResponse(BaseModel):
    status: str
    total: int
    success: int
    errors: int
    response: Optional[Dict[str, Any]] = None

class SearchRequest(BaseModel):
    user_id: str
    query_text: str
    k: int = 5
    min_score: Optional[float] = None
    filter_metadata: Optional[Dict[str, Any]] = None

class SearchResponse(BaseModel):
    results: List[Dict[str, Any]]
    total: int

# Agent Models
class AgentRequest(BaseModel):
    prompt: str = Field(..., description="User query or command")
    user_id: Optional[str] = Field("default_user", description="User identifier for task/RAG isolation")
    github_pat: Optional[str] = Field(None, description="GitHub Personal Access Token for MCP operations")

    class Config:
        schema_extra = {
            "example": {
                "prompt": "Create a task to learn Python",
                "user_id": "user123",
                "github_pat": "ghp_xxx"
            }
        }

class AgentResponse(BaseModel):
    result: str
    user_id: str
    agent_system: str = "Multi-Agent Supervisor"
    available_agents: list = ["Task Agent", "RAG Agent", "MCP Agent"]

class AgentInfo(BaseModel):
    name: str
    capabilities: list
    requires: list
    status: str

class AgentsListResponse(BaseModel):
    agents: List[AgentInfo]

@app.get("/")
async def root():
    """Root endpoint with API information"""
    return {
        "message": "ATOK AI Agents API",
        "version": "0.1.0",
        "endpoints": {
            "transcribe": "/transcribe - POST - Upload audio file for transcription",
            "enhance": "/enhance - POST - Enhance transcribed text",
            "transcribe_enhance": "/transcribe-enhance - POST - Transcribe and enhance audio in one call",
            "agent": {
                "invoke": "/agent/invoke - POST - Interact with multi-agent system",
                "stream": "/agent/stream - POST - Streaming agent responses (SSE)",
                "agents": "/agent/agents - GET - List available agents"
            },
            "health": "/health - GET - Health check",
            "opensearch": {
                "check": "/opensearch/collection/check/{user_id} - GET - Check if user collection exists",
                "create": "/opensearch/collection/create - POST - Create user collection",
                "insert": "/opensearch/document/insert - POST - Insert document to user collection",
                "bulk_insert": "/opensearch/document/bulk - POST - Bulk insert documents",
                "search": "/opensearch/search - POST - Similarity search in user collection",
                "stats": "/opensearch/collection/stats/{user_id} - GET - Get collection statistics",
                "delete": "/opensearch/collection/delete/{user_id} - DELETE - Delete user collection"
            }
        }
    }

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    # Check core services
    services = {
        "api": "healthy",
        "database": "unknown",
        "opensearch": "unknown",
        "transcription": "unknown"
    }
    
    # Check database
    try:
        from database import get_db_connection
        conn = get_db_connection()
        conn.close()
        services["database"] = "healthy"
    except:
        services["database"] = "unavailable"
    
    # Check OpenSearch config
    if os.getenv("OPENSEARCH_HOST"):
        services["opensearch"] = "configured"
    else:
        services["opensearch"] = "not_configured"
    
    # Check AWS Transcribe config
    if os.getenv("AWS_ACCESS_KEY_ID") and os.getenv("AWS_SECRET_ACCESS_KEY") and os.getenv("S3_BUCKET_NAME"):
        services["transcription"] = "configured"
    else:
        services["transcription"] = "not_configured"
    
    # Overall status
    core_healthy = services["api"] == "healthy" and services["database"] == "healthy"
    
    return {
        "status": "healthy" if core_healthy else "degraded",
        "services": services,
        "version": "0.2.0",
        "features": {
            "agent_system": "available",
            "task_management": "available" if services["database"] == "healthy" else "unavailable",
            "knowledge_search": "available" if services["opensearch"] == "configured" else "unavailable",
            "transcription": "available" if services["transcription"] == "configured" else "unavailable"
        }
    }

@app.post("/transcribe", response_model=TranscribeResponse)
async def transcribe_audio(
    file: UploadFile = File(...),
    bucket_name: Optional[str] = Form(None),
    api_key: str = Depends(verify_api_key)
):
    """
    Transcribe audio file using Amazon Transcribe
    
    Args:
        file: Audio file to transcribe (mp3, wav, flac, m4a, etc.)
        bucket_name: S3 bucket name (optional, will use env var if not provided)
    
    Returns:
        TranscribeResponse with transcription results
    
    Security:
        Requires X-API-Key header
    """
    
    # Validate file type
    if not file.content_type or not file.content_type.startswith('audio/'):
        raise HTTPException(status_code=400, detail="File must be an audio file")
    
    # Get bucket name from form or environment
    if not bucket_name:
        bucket_name = os.getenv('S3_BUCKET_NAME')
    
    if not bucket_name:
        raise HTTPException(
            status_code=400, 
            detail="S3_BUCKET_NAME must be provided or set in environment variables"
        )
    
    # Check AWS credentials
    if not os.getenv('AWS_ACCESS_KEY_ID') or not os.getenv('AWS_SECRET_ACCESS_KEY'):
        raise HTTPException(
            status_code=400,
            detail="AWS_ACCESS_KEY_ID and AWS_SECRET_ACCESS_KEY must be set in environment variables"
        )
    
    # Create temporary file
    temp_file = None
    try:
        # Create temporary file with proper extension
        file_extension = os.path.splitext(file.filename)[1] if file.filename else '.mp3'
        with tempfile.NamedTemporaryFile(delete=False, suffix=file_extension) as temp_file:
            content = await file.read()
            temp_file.write(content)
            temp_file_path = temp_file.name
        
        # Call transcription function
        result = await transcribe_audio_file_api(temp_file_path, bucket_name)
        
        # Extract only clean data
        detected_languages = []
        if result.get("detected_languages"):
            for lang in result["detected_languages"]:
                detected_languages.append({
                    "language_code": lang.get("language_code"),
                    "duration": f"{lang.get('duration_in_seconds', 0):.1f}s"
                })
        
        return TranscribeResponse(
            success=True,
            transcript=result.get("transcript"),
            detected_languages=detected_languages if detected_languages else None
        )
        
    except Exception as e:
        return TranscribeResponse(
            success=False,
            error=str(e)
        )
        
    finally:
        # Clean up temporary file
        if temp_file and os.path.exists(temp_file_path):
            os.unlink(temp_file_path)

@app.post("/enhance", response_model=EnhanceResponse)
async def enhance_text(
    request: EnhanceRequest,
    api_key: str = Depends(verify_api_key)
):
    """
    Enhance transcribed text using AI
    
    Args:
        request: EnhanceRequest with text to enhance and optional context
    
    Returns:
        EnhanceResponse with enhanced text
    
    Security:
        Requires X-API-Key header
    """
    
    try:
        # Create context-aware prompt
        context_info = f"\nContext: {request.context}" if request.context else ""
        
        input_message = HumanMessage(
            content=f"""
            The following is a transcription that may contain errors or inaccuracies.
            Please enhance and correct it while maintaining the original meaning.
            {context_info}
            
            Transcription to enhance:
            {request.text}
            """
        )
        
        # Get enhanced text from LLM
        response = llm.invoke([system_prompt, input_message])
        
        # Extract text from response (handle both string and list formats)
        if isinstance(response.content, str):
            enhanced_text = response.content
        elif isinstance(response.content, list):
            # Extract text from list format (reasoning content + text)
            text_parts = []
            for item in response.content:
                if isinstance(item, dict) and item.get('type') == 'text':
                    text_parts.append(item.get('text', ''))
            enhanced_text = '\n'.join(text_parts) if text_parts else str(response.content)
        else:
            enhanced_text = str(response.content)
        
        return EnhanceResponse(
            success=True,
            original_text=request.text,
            enhanced_text=enhanced_text
        )
        
    except Exception as e:
        return EnhanceResponse(
            success=False,
            original_text=request.text,
            error=str(e)
        )

@app.post("/transcribe-enhance", response_model=TranscribeEnhanceResponse)
async def transcribe_and_enhance(
    file: UploadFile = File(...),
    context: Optional[str] = Form(None),
    bucket_name: Optional[str] = Form(None),
    api_key: str = Depends(verify_api_key)
):
    """
    Combined endpoint: Transcribe audio file and enhance the transcript in one call
    
    Args:
        file: Audio file to transcribe (mp3, wav, flac, m4a, etc.)
        context: Optional context for enhancement (e.g., "meeting notes", "interview")
        bucket_name: S3 bucket name (optional, will use env var if not provided)
    
    Returns:
        TranscribeEnhanceResponse with both transcript and enhanced text
    
    Security:
        Requires X-API-Key header
    """
    
    # Validate file type
    if not file.content_type or not file.content_type.startswith('audio/'):
        raise HTTPException(status_code=400, detail="File must be an audio file")
    
    # Get bucket name from form or environment
    if not bucket_name:
        bucket_name = os.getenv('S3_BUCKET_NAME')
    
    if not bucket_name:
        raise HTTPException(
            status_code=400, 
            detail="S3_BUCKET_NAME must be provided or set in environment variables"
        )
    
    # Check AWS credentials
    if not os.getenv('AWS_ACCESS_KEY_ID') or not os.getenv('AWS_SECRET_ACCESS_KEY'):
        raise HTTPException(
            status_code=400,
            detail="AWS_ACCESS_KEY_ID and AWS_SECRET_ACCESS_KEY must be set in environment variables"
        )
    
    temp_file = None
    try:
        # Step 1: Transcribe audio
        file_extension = os.path.splitext(file.filename)[1] if file.filename else '.mp3'
        with tempfile.NamedTemporaryFile(delete=False, suffix=file_extension) as temp_file:
            content = await file.read()
            temp_file.write(content)
            temp_file_path = temp_file.name
        
        # Call transcription function
        transcribe_result = await transcribe_audio_file_api(temp_file_path, bucket_name)
        
        # Extract transcript
        transcript = transcribe_result.get("transcript")
        if not transcript:
            return TranscribeEnhanceResponse(
                success=False,
                error="Transcription failed: No transcript generated"
            )
        
        # Extract detected languages
        detected_languages = []
        if transcribe_result.get("detected_languages"):
            for lang in transcribe_result["detected_languages"]:
                detected_languages.append({
                    "language_code": lang.get("language_code"),
                    "duration": f"{lang.get('duration_in_seconds', 0):.1f}s"
                })
        
        # Step 2: Enhance the transcript
        try:
            context_info = f"\nContext: {context}" if context else ""
            
            input_message = HumanMessage(
                content=f"""
                The following is a transcription that may contain errors or inaccuracies.
                Please enhance and correct it while maintaining the original meaning.
                {context_info}
                
                Transcription to enhance:
                {transcript}
                """
            )
            
            # Get enhanced text from LLM
            llm_response = llm.invoke([system_prompt, input_message])
            
            # Extract text from response
            if isinstance(llm_response.content, str):
                enhanced_text = llm_response.content
            elif isinstance(llm_response.content, list):
                text_parts = []
                for item in llm_response.content:
                    if isinstance(item, dict) and item.get('type') == 'text':
                        text_parts.append(item.get('text', ''))
                enhanced_text = '\n'.join(text_parts) if text_parts else str(llm_response.content)
            else:
                enhanced_text = str(llm_response.content)
            
            return TranscribeEnhanceResponse(
                success=True,
                transcript=transcript,
                enhanced_text=enhanced_text,
                detected_languages=detected_languages if detected_languages else None
            )
            
        except Exception as enhance_error:
            # If enhancement fails, still return the transcript
            return TranscribeEnhanceResponse(
                success=True,
                transcript=transcript,
                enhanced_text=None,
                detected_languages=detected_languages if detected_languages else None,
                error=f"Enhancement failed: {str(enhance_error)}"
            )
        
    except Exception as e:
        return TranscribeEnhanceResponse(
            success=False,
            error=str(e)
        )
        
    finally:
        # Clean up temporary file
        if temp_file and os.path.exists(temp_file_path):
            os.unlink(temp_file_path)

# ============================================
# Agent Endpoints
# ============================================

@app.post("/agent/invoke", response_model=AgentResponse)
async def invoke_agent(
    request: AgentRequest,
    api_key: str = Depends(verify_api_key)
):
    """
    Interact with the multi-agent system (non-streaming)
    
    The supervisor agent will route your request to the appropriate specialist:
    - **Task Agent**: Create, read, complete, delete tasks
    - **RAG Agent**: Search knowledge base, answer questions
    - **MCP Agent**: GitHub operations (repos, issues, PRs)
    
    Args:
        prompt: Your query or command (required)
        user_id: User identifier for task/RAG isolation (optional, default: "default_user")
        github_pat: GitHub PAT for MCP operations (optional)
    
    Examples:
        - Task: {"prompt": "Create a task to learn Python", "user_id": "user123"}
        - RAG: {"prompt": "What did we discuss in meeting 3?", "user_id": "user123"}
        - MCP: {"prompt": "Show my GitHub repos", "github_pat": "ghp_xxx"}
    
    Security:
        Requires X-API-Key header
    """
    try:
        # Auto-fetch GitHub PAT from database if not provided
        github_pat = request.github_pat
        if not github_pat and request.user_id != "default_user":
            # Try to get GitHub token from database
            db_token = get_github_token(request.user_id)
            if db_token:
                github_pat = db_token
                print(f"✓ Auto-fetched GitHub PAT for user: {request.user_id}")
        
        # Set GitHub PAT in environment if available
        if github_pat:
            os.environ['GITHUB_PAT'] = github_pat
        
        # Add user context to the message
        contextualized_message = f"[User: {request.user_id}] {request.prompt}"
        
        # Call the supervisor agent
        result = supervisor(contextualized_message)
        
        return AgentResponse(
            result=str(result),
            user_id=request.user_id
        )
        
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Agent processing failed: {str(e)}"
        )

@app.post("/agent/stream")
async def stream_agent(
    request: AgentRequest,
    api_key: str = Depends(verify_api_key)
):
    """
    Streaming endpoint for real-time agent responses with Server-Sent Events (SSE)
    
    Returns real-time updates including:
    - Tool usage logs
    - Agent thinking process
    - Text generation word-by-word
    - Final result
    
    Args:
        prompt: Your query or command (required)
        user_id: User identifier for task/RAG isolation (optional)
        github_pat: GitHub PAT for MCP operations (optional)
    
    Example usage with curl:
    ```bash
    curl -X POST http://localhost:8000/agent/stream \
      -H "X-API-Key: your-api-key" \
      -H "Content-Type: application/json" \
      -d '{"prompt": "Create a task", "user_id": "user123"}' \
      --no-buffer
    ```
    
    Security:
        Requires X-API-Key header
    """
    import asyncio
    import sys
    import io
    from queue import Queue, Empty
    from threading import Thread
    from strands import Agent
    from agents.supervisor import task_assistant, rag_assistant, mcp_assistant, SUPERVISOR_PROMPT, supervisor as supervisor_agent
    
    async def event_generator():
        try:
            # Auto-fetch GitHub PAT from database if not provided
            github_pat = request.github_pat
            if not github_pat and request.user_id != "default_user":
                db_token = get_github_token(request.user_id)
                if db_token:
                    github_pat = db_token
                    yield f"data: {json.dumps({'type': 'info', 'message': 'Auto-fetched GitHub PAT'})}\n\n"
            
            if github_pat:
                os.environ['GITHUB_PAT'] = github_pat
            
            contextualized_message = f"[User: {request.user_id}] {request.prompt}"
            
            # Send initial event
            yield f"data: {json.dumps({'type': 'init', 'user_id': request.user_id, 'message': 'Starting agent...'})}\n\n"
            
            # Event queue
            event_queue = Queue()
            is_done = {'value': False}
            
            # Capture stdout
            class StreamCapture(io.StringIO):
                def write(self, text):
                    if text and text.strip():
                        event_queue.put({'type': 'stdout', 'data': text}, block=False)
                    return super().write(text)
            
            # Callback handler - SIMPLE VERSION
            def handle_events(**kwargs):
                try:
                    if kwargs:
                        event_queue.put(kwargs, block=False)
                except:
                    pass
            
            # Use the existing supervisor agent instead of creating new one
            # Just set the callback handler
            streaming_supervisor = supervisor_agent
            streaming_supervisor.callback_handler = handle_events
            
            # Result container
            result_container = {'result': None, 'error': None}
            
            # Run agent in thread with stdout capture
            def run_agent():
                old_stdout = sys.stdout
                sys.stdout = StreamCapture()
                try:
                    result = streaming_supervisor(contextualized_message)
                    result_container['result'] = str(result) if result else "No result"
                except Exception as e:
                    # Print full traceback to console for debugging
                    import traceback
                    traceback.print_exc()
                    result_container['error'] = str(e)
                finally:
                    sys.stdout = old_stdout
                    is_done['value'] = True
                    event_queue.put({'_done': True}, block=False)
            
            agent_thread = Thread(target=run_agent, daemon=True)
            agent_thread.start()
            
            # Stream events - SIMPLE VERSION
            last_event_time = asyncio.get_event_loop().time()
            while not is_done['value'] or not event_queue.empty():
                try:
                    event = event_queue.get(block=True, timeout=0.05)
                    
                    # Skip if event is None
                    if event is None:
                        continue
                    
                    if event.get('_done'):
                        break
                    
                    event_data = None
                    
                    # Wrap all event processing in try-catch
                    try:
                        # Process stdout capture
                        if event.get('type') == 'stdout':
                            event_data = {
                                "type": "log",
                                "data": event.get('data', ''),
                                "message": str(event.get('data', '')).strip()
                            }
                        # Process callback events
                        elif event.get("init_event_loop"):
                            event_data = {"type": "init_loop", "message": "🔄 Event loop initialized"}
                        elif event.get("start_event_loop"):
                            event_data = {"type": "start", "message": "▶️ Processing..."}
                        elif event.get("complete"):
                            event_data = {"type": "complete", "message": "✅ Cycle completed"}
                        elif "current_tool_use" in event:
                            # Safe access to tool name
                            tool_use = event.get("current_tool_use")
                            if tool_use and isinstance(tool_use, dict):
                                tool_name = tool_use.get("name")
                                if tool_name:
                                    event_data = {
                                        "type": "tool",
                                        "tool_name": tool_name,
                                        "message": f"🔧 Using tool: {tool_name}"
                                    }
                        elif "data" in event:
                            data_value = event.get("data")
                            if data_value is not None:
                                event_data = {
                                    "type": "text",
                                    "data": str(data_value)
                                }
                    except Exception as parse_error:
                        # Skip malformed events
                        print(f"Skipping malformed event: {parse_error}")
                        continue
                    
                    if event_data:
                        yield f"data: {json.dumps(event_data)}\n\n"
                        last_event_time = asyncio.get_event_loop().time()
                        
                except Empty:
                    # Send keepalive every 2 seconds
                    current_time = asyncio.get_event_loop().time()
                    if current_time - last_event_time > 2:
                        yield f": keepalive\n\n"
                        last_event_time = current_time
                    await asyncio.sleep(0.01)
                except Exception as e:
                    print(f"Event processing error: {e}")
                    await asyncio.sleep(0.01)
            
            # Wait for thread
            agent_thread.join(timeout=5)
            
            # Send final result
            if result_container['result']:
                yield f"data: {json.dumps({'type': 'result', 'result': result_container['result'], 'user_id': request.user_id})}\n\n"
            elif result_container['error']:
                yield f"data: {json.dumps({'type': 'error', 'error': result_container['error'], 'user_id': request.user_id})}\n\n"
            
            yield f"data: {json.dumps({'type': 'done', 'message': '🏁 Stream completed'})}\n\n"
            
        except Exception as e:
            yield f"data: {json.dumps({'type': 'error', 'error': str(e), 'user_id': request.user_id})}\n\n"
    
    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
            "Transfer-Encoding": "chunked"
        }
    )

@app.get("/agent/agents", response_model=AgentsListResponse)
async def list_agents(api_key: str = Depends(verify_api_key)):
    """
    List available agents and their capabilities
    
    Returns information about:
    - Task Agent: Task management operations
    - RAG Agent: Knowledge retrieval and search
    - MCP Agent: GitHub operations
    
    Security:
        Requires X-API-Key header
    """
    return AgentsListResponse(
        agents=[
            AgentInfo(
                name="Task Agent",
                capabilities=[
                    "Create tasks",
                    "Read tasks",
                    "Complete tasks",
                    "Delete tasks"
                ],
                requires=["user_id"],
                status="ready"
            ),
            AgentInfo(
                name="RAG Agent",
                capabilities=[
                    "Semantic search",
                    "Knowledge retrieval",
                    "Question answering"
                ],
                requires=["user_id", "OpenSearch"],
                status="ready"
            ),
            AgentInfo(
                name="MCP Agent",
                capabilities=[
                    "GitHub repository operations",
                    "Issue management",
                    "Pull request operations",
                    "File operations"
                ],
                requires=["github_pat"],
                status="ready"
            )
        ]
    )

# ============================================
# OpenSearch Endpoints
# ============================================

@app.get("/opensearch/collection/check/{user_id}", response_model=CollectionCheckResponse)
async def check_collection(
    user_id: str,
    api_key: str = Depends(verify_api_key)
):
    """
    Check apakah collection untuk user sudah ada
    
    Args:
        user_id: ID user (contoh: user69, user_211123)
    
    Returns:
        exists: True/False
        index_name: nama index
        message: status message
        stats: statistics jika collection ada
    
    Security:
        Requires X-API-Key header
    """
    try:
        client = get_opensearch_client()
        index_name = f"user_{user_id}_collection"
        
        # Check if index exists
        exists = client.client.indices.exists(index=index_name)
        
        if exists:
            # Get stats if exists
            try:
                stats = client.get_collection_stats(user_id)
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

@app.post("/opensearch/collection/create", response_model=CollectionCreateResponse)
async def create_collection(
    request: CollectionCreateRequest,
    api_key: str = Depends(verify_api_key)
):
    """
    Buat collection baru untuk user
    
    Args:
        user_id: ID user (REQUIRED)
        vector_dimension: dimensi vector (default: 1536 untuk Cohere v4)
    
    Returns:
        status: created/exists
        message: status message
        index_name: nama index
    
    Security:
        Requires X-API-Key header
    """
    try:
        client = get_opensearch_client()
        result = client.create_user_collection(
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

@app.post("/opensearch/document/insert", response_model=DocumentInsertResponse)
async def insert_document(
    request: DocumentInsertRequest,
    api_key: str = Depends(verify_api_key)
):
    """
    Insert document (text + vector) ke user collection
    
    Args:
        user_id: ID user (REQUIRED)
        text: Text content to insert
        metadata: Optional metadata dictionary
        doc_id: Optional document ID
    
    Returns:
        status: success
        index_name: nama index
        doc_id: ID dokumen yang diinsert
    
    Security:
        Requires X-API-Key header
    """
    try:
        client = get_opensearch_client()
        result = client.insert_document(
            user_id=request.user_id,
            text=request.text,
            metadata=request.metadata,
            doc_id=request.doc_id
        )
        
        return DocumentInsertResponse(
            status=result['status'],
            index_name=result['index_name'],
            doc_id=result['doc_id'],
            response=result.get('response')
        )
        
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error inserting document: {str(e)}")

@app.post("/opensearch/document/bulk", response_model=BulkInsertResponse)
async def bulk_insert_documents(
    request: BulkInsertRequest,
    api_key: str = Depends(verify_api_key)
):
    """
    Bulk insert multiple documents ke user collection
    
    Args:
        user_id: ID user (REQUIRED)
        documents: List of documents with 'text', optional 'metadata' and 'doc_id'
    
    Returns:
        status: completed
        total: total documents processed
        success: successful inserts
        errors: failed inserts
    
    Security:
        Requires X-API-Key header
    """
    try:
        client = get_opensearch_client()
        result = client.bulk_insert_documents(
            user_id=request.user_id,
            documents=request.documents
        )
        
        return BulkInsertResponse(
            status=result['status'],
            total=result['total'],
            success=result['success'],
            errors=result['errors'],
            response=result.get('response')
        )
        
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error bulk inserting documents: {str(e)}")

@app.post("/opensearch/search", response_model=SearchResponse)
async def similarity_search(
    request: SearchRequest,
    api_key: str = Depends(verify_api_key)
):
    """
    Similarity search menggunakan vector similarity
    
    Args:
        user_id: ID user (REQUIRED)
        query_text: Text query untuk search
        k: Number of results to return (default: 5)
        min_score: Minimum similarity score (optional)
        filter_metadata: Filter by metadata (optional)
    
    Returns:
        results: List of matching documents with scores
        total: Total results returned
    
    Security:
        Requires X-API-Key header
    """
    try:
        client = get_opensearch_client()
        results = client.similarity_search(
            user_id=request.user_id,
            query_text=request.query_text,
            k=request.k,
            min_score=request.min_score,
            filter_metadata=request.filter_metadata
        )
        
        return SearchResponse(
            results=results,
            total=len(results)
        )
        
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error searching: {str(e)}")

@app.get("/opensearch/collection/stats/{user_id}")
async def get_collection_stats(
    user_id: str,
    api_key: str = Depends(verify_api_key)
):
    """
    Get statistics dari collection user
    
    Args:
        user_id: ID user (REQUIRED)
    
    Returns:
        index_name: nama index
        document_count: jumlah dokumen
        size_in_bytes: ukuran collection
        stats: detailed statistics
    
    Security:
        Requires X-API-Key header
    """
    try:
        client = get_opensearch_client()
        stats = client.get_collection_stats(user_id)
        return stats
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error getting stats: {str(e)}")

@app.delete("/opensearch/collection/delete/{user_id}")
async def delete_collection(
    user_id: str,
    api_key: str = Depends(verify_api_key)
):
    """
    Hapus collection user (untuk testing/cleanup)
    
    Args:
        user_id: ID user (REQUIRED)
    
    Returns:
        status: deleted/not_found
        message: status message
    
    Security:
        Requires X-API-Key header
    """
    try:
        client = get_opensearch_client()
        result = client.delete_user_collection(user_id)
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error deleting collection: {str(e)}")

# Modified transcription function for API use
async def transcribe_audio_file_api(audio_file_path: str, bucket_name: str):
    """
    Modified version of transcribe_audio_file for API use
    Returns structured data instead of printing to console
    """
    import boto3
    import time
    import urllib.request
    import asyncio
    
    # Get AWS credentials from environment variables
    aws_access_key = os.getenv('AWS_ACCESS_KEY_ID')
    aws_secret_key = os.getenv('AWS_SECRET_ACCESS_KEY')
    aws_region = os.getenv('AWS_REGION', 'ap-southeast-1')
    
    if not aws_access_key or not aws_secret_key:
        raise ValueError("AWS_ACCESS_KEY_ID and AWS_SECRET_ACCESS_KEY must be set in environment variables")
    
    # Initialize AWS clients
    s3_client = boto3.client(
        's3',
        aws_access_key_id=aws_access_key,
        aws_secret_access_key=aws_secret_key,
        region_name=aws_region
    )
    
    transcribe_client = boto3.client(
        'transcribe',
        aws_access_key_id=aws_access_key,
        aws_secret_access_key=aws_secret_key,
        region_name=aws_region
    )
    
    # Create unique job name and S3 key
    timestamp = int(time.time())
    job_name = f"transcribe-job-{timestamp}"
    
    # Get file name and create S3 key
    file_name = os.path.basename(audio_file_path)
    s3_key = f"transcribe/audio/{timestamp}/{file_name}"
    
    # Detect media format from file extension
    file_extension = os.path.splitext(audio_file_path)[1].lower().lstrip('.')
    supported_formats = ['mp3', 'mp4', 'wav', 'flac', 'ogg', 'amr', 'webm', 'm4a']
    
    if file_extension not in supported_formats:
        raise ValueError(f"Unsupported audio format: {file_extension}. Supported formats: {', '.join(supported_formats)}")
    
    media_format = file_extension
    
    try:
        # Upload file to S3
        s3_client.upload_file(audio_file_path, bucket_name, s3_key)
        
        # Get the S3 URI
        media_uri = f"s3://{bucket_name}/{s3_key}"
        
        # Define language options for automatic identification
        language_options = [
            'id-ID', 
            'en-US',  
            'es-US', 
            'fr-FR',  
            'de-DE',  
            'it-IT',  
            'pt-BR',  
            'ja-JP',  
            'ko-KR', 
            'zh-CN',  
        ]
        
        # Start transcription job
        response = transcribe_client.start_transcription_job(
            TranscriptionJobName=job_name,
            Media={'MediaFileUri': media_uri},
            MediaFormat=media_format,
            IdentifyMultipleLanguages=True,
            LanguageOptions=language_options,
            OutputBucketName=bucket_name,
            OutputKey=f"transcribe/output/{timestamp}/",
        )
        
        # Wait for job to complete
        while True:
            status = transcribe_client.get_transcription_job(TranscriptionJobName=job_name)
            job_status = status['TranscriptionJob']['TranscriptionJobStatus']
            
            if job_status in ['COMPLETED', 'FAILED']:
                break
            
            await asyncio.sleep(5)
        
        if job_status == 'COMPLETED':
            # Get the transcription results
            transcript_uri = status['TranscriptionJob']['Transcript']['TranscriptFileUri']
            
            # Download and parse the results
            with urllib.request.urlopen(transcript_uri) as response:
                transcript_data = json.loads(response.read().decode('utf-8'))
            
            # Extract results
            detected_languages = []
            if 'language_codes' in transcript_data['results']:
                detected_languages = transcript_data['results']['language_codes']
            
            full_transcript = ""
            if 'transcripts' in transcript_data['results']:
                full_transcript = transcript_data['results']['transcripts'][0]['transcript']
            
            return {
                "job_name": job_name,
                "transcript": full_transcript,
                "detected_languages": detected_languages,
                "detailed_results": transcript_data
            }
        else:
            failure_reason = status['TranscriptionJob'].get('FailureReason', 'Unknown error')
            raise Exception(f"Transcription failed: {failure_reason}")
    
    finally:
        # Cleanup: Delete uploaded audio file
        try:
            s3_client.delete_object(Bucket=bucket_name, Key=s3_key)
        except Exception:
            pass  # Ignore cleanup errors
        
        # Delete the transcription job
        try:
            transcribe_client.delete_transcription_job(TranscriptionJobName=job_name)
        except Exception:
            pass  # Ignore cleanup errors

# Lambda handler (untuk AWS Lambda)
from mangum import Mangum
handler = Mangum(app)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
