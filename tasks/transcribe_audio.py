import boto3
import time
from datetime import datetime
from config.settings import AWS_REGION, S3_BUCKET_NAME
from utils.s3_helper import get_s3_file_url, list_s3_files
from dotenv import load_dotenv
import os

load_dotenv()
def transcribe_audio_with_aws():
    aws_key = os.getenv("AWS_ACCESS_KEY")
    aws_secret = os.getenv("AWS_SECRET_KEY")
    
    if not aws_key or not aws_secret:
        raise ValueError("AWS credentials are missing! Check your .env file.")
    """Fetches latest audio file from S3, transcribes it, and saves the result to another S3 bucket."""
    
    # Initialize AWS Transcribe & S3 clients
    transcribe_client = boto3.client("transcribe", region_name=AWS_REGION)

    # Get the latest audio file from S3 bucket dynamically
    audio_files = list_s3_files(S3_BUCKET_NAME, prefix="audio/")  # Fetch all audio files in `audio/` folder
    if not audio_files:
        print("❌ No audio files found in S3 bucket.")
        return

    latest_audio_file = max(audio_files, key=lambda f: f["LastModified"])["Key"]  # Get latest file
    file_url = get_s3_file_url(S3_BUCKET_NAME, latest_audio_file)
    
    # Generate unique transcription job name
    job_name = f"transcribe_job_{int(datetime.now().timestamp())}"

    # Start transcription job
    print(f"🚀 Starting transcription for {latest_audio_file}...")
    transcribe_client.start_transcription_job(
        TranscriptionJobName=job_name,
        Media={'MediaFileUri': file_url},
        MediaFormat='mp3',
        IdentifyLanguage=True,  # Auto-detect language
        OutputBucketName=S3_BUCKET_NAME,  # Save transcription to the same S3 bucket
        OutputKey=f"transcriptions/{latest_audio_file.replace('.mp3', '.json')}",  # Save JSON result
        Settings={
            "ShowSpeakerLabels": True,
            "MaxSpeakerLabels": 3
        }
    )

    # Wait for transcription job to complete
    while True:
        response = transcribe_client.get_transcription_job(TranscriptionJobName=job_name)
        status = response["TranscriptionJob"]["TranscriptionJobStatus"]
        if status in ["COMPLETED", "FAILED"]:
            break
        print("⏳ Waiting for transcription to complete...")
        time.sleep(10)

    # Get and print the transcript URL if successful
    if status == "COMPLETED":
        transcript_uri = response["TranscriptionJob"]["Transcript"]["TranscriptFileUri"]
        print(f"✅ Transcription completed. Download transcript: {transcript_uri}")
    else:
        print("❌ Transcription job failed.")

