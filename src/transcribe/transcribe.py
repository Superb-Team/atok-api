import asyncio
import os
import json
from io import BytesIO
import soundfile as sf
from scipy import signal
import boto3
from botocore.exceptions import ClientError
from dotenv import load_dotenv
load_dotenv()


async def transcribe_audio_file(audio_file_path: str, bucket_name: str):
    """
    Transcribe audio file using Amazon Transcribe (batch mode) with automatic language identification
    
    Args:
        audio_file_path: Path to the audio file (e.g., test.mp3)
        bucket_name: Name of your existing S3 bucket
    """
    # Get AWS credentials from environment variables
    aws_access_key = os.getenv('AWS_ACCESS_KEY_ID')
    aws_secret_key = os.getenv('AWS_SECRET_ACCESS_KEY')
    aws_region = os.getenv('AWS_REGION', 'ap-southeast-1')
    
    if not aws_access_key or not aws_secret_key:
        raise ValueError("AWS_ACCESS_KEY_ID and AWS_SECRET_ACCESS_KEY must be set in environment variables")
    
    print(f"Using AWS Region: {aws_region}")
    print(f"Using S3 Bucket: {bucket_name}")
    print(f"Processing audio file: {audio_file_path}\n")
    
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
    import time
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
    print(f"Detected audio format: {media_format}")
    
    try:
        # Upload file to S3
        print(f"Uploading {audio_file_path} to S3...")
        s3_client.upload_file(audio_file_path, bucket_name, s3_key)
        
        # Get the S3 URI
        media_uri = f"s3://{bucket_name}/{s3_key}"
        print(f"✅ File uploaded to: {media_uri}\n")
        
        # Define language options for automatic identification
        language_options = [
            'id-ID',  # Indonesian
            'en-US',  # English (US)
            'es-US',  # Spanish (US)
            'fr-FR',  # French
            'de-DE',  # German
            'it-IT',  # Italian
            'pt-BR',  # Portuguese (Brazil)
            'ja-JP',  # Japanese
            'ko-KR',  # Korean
            'zh-CN',  # Chinese (Mandarin)
        ]
        
        print("Starting transcription job with automatic language identification...")
        print(f"Supported languages: {', '.join(language_options)}\n")
        
        # Start transcription job
        response = transcribe_client.start_transcription_job(
            TranscriptionJobName=job_name,
            Media={'MediaFileUri': media_uri},
            MediaFormat=media_format,
            IdentifyMultipleLanguages=True,
            LanguageOptions=language_options,
            # Output to the same bucket
            OutputBucketName=bucket_name,
            OutputKey=f"transcribe/output/{timestamp}/",
        )
        
        print(f"Transcription job '{job_name}' started. Waiting for completion...")
        
        # Wait for job to complete
        while True:
            status = transcribe_client.get_transcription_job(TranscriptionJobName=job_name)
            job_status = status['TranscriptionJob']['TranscriptionJobStatus']
            
            if job_status in ['COMPLETED', 'FAILED']:
                break
            
            print(f"Job status: {job_status}... waiting 5 seconds")
            await asyncio.sleep(5)
        
        if job_status == 'COMPLETED':
            print(f"\nTranscription completed successfully!\n")
            
            # Get the transcription results
            transcript_uri = status['TranscriptionJob']['Transcript']['TranscriptFileUri']
            
            # Download and parse the results
            import urllib.request
            with urllib.request.urlopen(transcript_uri) as response:
                transcript_data = json.loads(response.read().decode('utf-8'))
            
            # Display results
            print("=" * 80)
            print("TRANSCRIPTION RESULTS")
            print("=" * 80)
            
            # Get detected language codes
            if 'language_codes' in transcript_data['results']:
                detected_languages = transcript_data['results']['language_codes']
                print(f"\nDetected Languages:")
                for lang in detected_languages:
                    print(f"   - {lang['language_code']} (confidence: {lang.get('duration_in_seconds', 'N/A')}s)")
            
            # Print full transcript
            if 'transcripts' in transcript_data['results']:
                full_transcript = transcript_data['results']['transcripts'][0]['transcript']
                print(f"\nFull Transcript:")
                print(f"   {full_transcript}")
            
            # Print detailed items with language codes
            print(f"\nDetailed Transcript with Language Detection:")
            if 'items' in transcript_data['results']:
                current_lang = None
                line = ""
                
                for item in transcript_data['results']['items']:
                    content = item.get('alternatives', [{}])[0].get('content', '')
                    item_type = item.get('type', 'pronunciation')
                    
                    # Check for language code in item
                    item_lang = None
                    if 'language_code' in item:
                        item_lang = item['language_code']
                    
                    # Print language header if language changed
                    if item_lang and item_lang != current_lang:
                        if line:
                            print(f"   {line}")
                            line = ""
                        print(f"\n   [{item_lang}]", end=" ")
                        current_lang = item_lang
                    
                    # Add content to line
                    if item_type == 'punctuation':
                        line += content
                    else:
                        line += f" {content}"
                
                if line:
                    print(f"   {line}")
            
            print("\n" + "=" * 80)
            
            # Save full results to file
            output_file = "transcription_results.json"
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(transcript_data, f, indent=2, ensure_ascii=False)
            print(f"\nFull results saved to: {output_file}")
            
        else:
            print(f"\nTranscription failed!")
            if 'FailureReason' in status['TranscriptionJob']:
                print(f"Reason: {status['TranscriptionJob']['FailureReason']}")
    
    finally:
        # Cleanup: Delete uploaded audio file (optional - uncomment if you want to keep it)
        print(f"\n🧹 Cleaning up...")
        try:
            # Delete the uploaded audio file
            s3_client.delete_object(Bucket=bucket_name, Key=s3_key)
            print(f"   Deleted uploaded audio: s3://{bucket_name}/{s3_key}")
            
            # Note: Transcription output will remain in S3 for your reference
            print(f"   transcription output kept in: s3://{bucket_name}/transcribe/output/{timestamp}/")
        except Exception as e:
            print(f"    Warning: Cleanup failed: {e}")
        
        # Delete the transcription job
        try:
            transcribe_client.delete_transcription_job(TranscriptionJobName=job_name)
            print(f"   Deleted transcription job: {job_name}")
        except Exception as e:
            print(f"   Warning: Could not delete job: {e}")


async def main():
    """
    Main function to run the transcription
    """
    audio_file = "data/test.wav"
    
    # Get bucket name from environment variable
    bucket_name = os.getenv('S3_BUCKET_NAME')
    
    if not bucket_name:
        print("Error: S3_BUCKET_NAME environment variable is not set!")
        print("\nPlease set it using:")
        print("  export S3_BUCKET_NAME='your-bucket-name'  # Linux/macOS")
        print("  set S3_BUCKET_NAME=your-bucket-name       # Windows CMD")
        print("  $env:S3_BUCKET_NAME='your-bucket-name'    # Windows PowerShell")
        return
    
    # Check if file exists
    if not os.path.exists(audio_file):
        print(f"Error: Audio file '{audio_file}' not found!")
        print("Please make sure the audio file exists in the data directory.")
        return
    
    try:
        await transcribe_audio_file(audio_file, bucket_name)
    except Exception as e:
        print(f"\nError during transcription: {str(e)}")
        import traceback
        traceback.print_exc()
        raise


if __name__ == "__main__":
    # Run the async main function
    asyncio.run(main())