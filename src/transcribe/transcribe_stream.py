import asyncio
import os
from amazon_transcribe.client import TranscribeStreamingClient
from amazon_transcribe.handlers import TranscriptResultStreamHandler
from amazon_transcribe.model import TranscriptEvent
import soundfile as sf
from scipy import signal
from dotenv import load_dotenv
from datetime import datetime
load_dotenv()


class ChunkedEventHandler(TranscriptResultStreamHandler):
    """
    Event handler dengan timestamp untuk chunking
    """
    def __init__(self, transcript_result_stream, chunk_duration=5):
        super().__init__(transcript_result_stream)
        self.chunk_duration = chunk_duration
        self.all_transcripts = []
        self.current_chunk_text = ""
        self.chunk_number = 0
        
    async def handle_transcript_event(self, transcript_event: TranscriptEvent):
        """Handle transcript events per chunk"""
        results = transcript_event.transcript.results
        
        for result in results:
            if not result.is_partial:
                # Final results
                for alt in result.alternatives:
                    transcript_text = alt.transcript.strip()
                    
                    if transcript_text:
                        timestamp = datetime.utcnow().strftime("%H:%M:%S")
                        
                        # Simpan ke list
                        self.all_transcripts.append({
                            'timestamp': timestamp,
                            'chunk': self.chunk_number,
                            'text': transcript_text
                        })
                        
                        # Print dengan format rapi
                        print(f"[{timestamp}] Chunk {self.chunk_number}: {transcript_text}")
                        self.current_chunk_text = transcript_text
                    
            else:
                # Partial results
                for alt in result.alternatives:
                    if alt.transcript:
                        timestamp = datetime.utcnow().strftime("%H:%M:%S")
                        print(f"\r[{timestamp}] ⏳ {alt.transcript}", end='', flush=True)


async def transcribe_chunked(audio_file_path: str, chunk_duration: int = 5):
    """
    Transcribe dengan chunking per X detik
    
    Args:
        audio_file_path: Path ke audio file
        chunk_duration: Durasi per chunk dalam detik (default: 5)
    """
    aws_access_key = os.getenv('AWS_ACCESS_KEY_ID')
    aws_secret_key = os.getenv('AWS_SECRET_ACCESS_KEY')
    aws_region = os.getenv('AWS_REGION', 'ap-southeast-1')
    
    if not aws_access_key or not aws_secret_key:
        raise ValueError("AWS credentials harus di-set di environment")
    
    os.environ['AWS_ACCESS_KEY_ID'] = aws_access_key
    os.environ['AWS_SECRET_ACCESS_KEY'] = aws_secret_key
    os.environ['AWS_DEFAULT_REGION'] = aws_region
    
    print("=" * 80)
    print(f"🎙️  CHUNKED TRANSCRIPTION (Every {chunk_duration} seconds)")
    print("=" * 80)
    print(f"🌐 Region: {aws_region}")
    print(f"🎵 Audio: {audio_file_path}")
    print(f"⏱️  Chunk Duration: {chunk_duration}s\n")
    
    # Load audio
    print("📂 Loading audio...")
    data, samplerate = sf.read(audio_file_path)
    
    # Convert ke mono
    if len(data.shape) > 1:
        data = data.mean(axis=1)
    
    # Resample ke 16kHz
    target_samplerate = 16000
    if samplerate != target_samplerate:
        print(f"🔄 Resampling {samplerate}Hz → {target_samplerate}Hz...")
        number_of_samples = round(len(data) * float(target_samplerate) / samplerate)
        data = signal.resample(data, number_of_samples)
        samplerate = target_samplerate
    
    # Normalize
    data = data / (abs(data).max() + 1e-10)
    pcm_data = (data * 32767).astype('int16')
    
    total_duration = len(pcm_data) / samplerate
    total_chunks = int(total_duration / chunk_duration) + 1
    
    print(f"✅ Audio ready: {total_duration:.1f}s total, {total_chunks} chunks\n")
    
    # Audio stream generator dengan chunking
    async def audio_stream_chunked():
        bytes_per_sample = 2
        samples_per_chunk = int(samplerate * chunk_duration)
        
        print("=" * 80)
        print("🎬 STARTING REAL-TIME TRANSCRIPTION")
        print("=" * 80 + "\n")
        
        chunk_num = 0
        for i in range(0, len(pcm_data), samples_per_chunk):
            chunk_num += 1
            chunk = pcm_data[i:i + samples_per_chunk].tobytes()
            
            # Marker untuk chunk baru
            print(f"\n{'─' * 40}")
            print(f"📦 Processing Chunk {chunk_num}/{total_chunks}")
            print(f"{'─' * 40}")
            
            yield chunk
            
            # Simulasi real-time: wait sesuai durasi chunk
            await asyncio.sleep(chunk_duration * 0.3)  # 30% speed untuk testing
        
        print(f"\n{'─' * 40}")
        print(f"✅ All chunks processed")
        print(f"{'─' * 40}\n")
    
    # Initialize Transcribe client
    client = TranscribeStreamingClient(region=aws_region)
    
    # Start stream
    stream = await client.start_stream_transcription(
        language_code="id-ID",
        media_sample_rate_hz=samplerate,
        media_encoding="pcm",
        enable_partial_results_stabilization=True,
        partial_results_stability="high",
    )
    
    # Handler
    handler = ChunkedEventHandler(stream.output_stream, chunk_duration)
    
    # Process
    await asyncio.gather(
        write_chunks(stream, audio_stream_chunked()),
        handler.handle_events()
    )
    
    # Summary
    print("\n" + "=" * 80)
    print("📊 TRANSCRIPTION SUMMARY")
    print("=" * 80)
    
    for item in handler.all_transcripts:
        print(f"[{item['timestamp']}] Chunk {item['chunk']}: {item['text']}")
    
    # Save
    output_file = f"transcription_chunked_{chunk_duration}s.txt"
    with open(output_file, 'w', encoding='utf-8') as f:
        for item in handler.all_transcripts:
            f.write(f"[{item['timestamp']}] {item['text']}\n")
    
    print(f"\n💾 Saved: {output_file}")
    print(f"✅ Total chunks processed: {len(handler.all_transcripts)}\n")


async def write_chunks(stream, audio_chunks):
    async for chunk in audio_chunks:
        await stream.input_stream.send_audio_event(audio_chunk=chunk)
    await stream.input_stream.end_stream()


async def main():
    audio_file = "data/test.mp3"
    
    if not os.path.exists(audio_file):
        print(f"❌ File not found: {audio_file}")
        return
    
    # Set chunk duration (dalam detik)
    chunk_duration = int(os.getenv('CHUNK_DURATION', '5'))  # Default 5 detik
    
    print(f"⚙️  Chunk duration set to: {chunk_duration} seconds\n")
    
    try:
        await transcribe_chunked(audio_file, chunk_duration)
    except Exception as e:
        print(f"\n❌ Error: {str(e)}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    asyncio.run(main())