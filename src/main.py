from fastapi import FastAPI, File, UploadFile, HTTPException, Form, Header, Depends
from fastapi.responses import JSONResponse
from fastapi.security import APIKeyHeader
from pydantic import BaseModel
import os
import tempfile
import json
from typing import Optional
from dotenv import load_dotenv

# Import our modules
from transcribe.transcribe import transcribe_audio_file
from enhance.enhance import llm, system_prompt
from langchain_core.messages import HumanMessage

# Load environment variables
load_dotenv()

app = FastAPI(
    title="ATOK AI Notetaker API Endpoint",
    description="API for audio transcription and text enhancement",
    version="0.0.5"
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

@app.get("/")
async def root():
    """Root endpoint with API information"""
    return {
        "message": "ATOK AI Agents API",
        "version": "1.0.0",
        "endpoints": {
            "transcribe": "/transcribe - POST - Upload audio file for transcription",
            "enhance": "/enhance - POST - Enhance transcribed text",
            "health": "/health - GET - Health check"
        }
    }

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    required_env_vars = ["AWS_ACCESS_KEY_ID", "AWS_SECRET_ACCESS_KEY", "S3_BUCKET_NAME"]
    missing_vars = [var for var in required_env_vars if not os.getenv(var)]
    
    return {
        "status": "healthy" if not missing_vars else "unhealthy",
        "missing_env_vars": missing_vars,
        "aws_region": os.getenv("AWS_REGION", "ap-southeast-1")
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
