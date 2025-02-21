import boto3
import time
from datetime import datetime
import os
from dotenv import load_dotenv
from utils.s3_helper import get_s3_file_url, list_s3_files

# Load environment variables
load_dotenv()

AWS_REGION = os.getenv("AWS_REGION")
S3_BUCKET_NAME = os.getenv("S3_BUCKET_NAME")

if not AWS_REGION or not S3_BUCKET_NAME:
    raise ValueError("Missing AWS_REGION or S3_BUCKET_NAME in .env file")

def transcribe_audio_with_aws():
    """Fetches the latest audio file from S3, transcribes it, and saves the result to another S3 bucket."""
    
    # Load AWS credentials
    aws_key = os.getenv("AWS_ACCESS_KEY")
    aws_secret = os.getenv("AWS_SECRET_KEY")

    if not aws_key or not aws_secret:
        raise ValueError("AWS credentials are missing! Check your .env file.")

    # Initialize AWS Transcribe & S3 clients
    transcribe_client = boto3.client("transcribe", region_name=AWS_REGION, 
                                     aws_access_key_id=aws_key, 
                                     aws_secret_access_key=aws_secret)

    # Get the latest audio file from S3 bucket dynamically
    audio_files = list_s3_files(S3_BUCKET_NAME, prefix="audio/")  
    
    if not audio_files:
        print("No audio files found in S3 bucket.")
        return

    # Get the most recent audio file
    latest_audio_file = max(audio_files, key=lambda f: f["LastModified"])["Key"]
    file_url = get_s3_file_url(S3_BUCKET_NAME, latest_audio_file)

    # Generate unique transcription job name and output file key
    timestamp = int(datetime.now().timestamp())
    job_name = f"transcribe_job_{timestamp}"
    output_file_key = f"transcriptions/{latest_audio_file.replace('.mp3', '')}_{timestamp}.json"

    # Start transcription job
    print(f"Starting transcription for {latest_audio_file}...")
    
    try:
        transcribe_client.start_transcription_job(
            TranscriptionJobName=job_name,
            Media={'MediaFileUri': file_url},
            MediaFormat='mp3',
            IdentifyLanguage=True,  # Auto-detect language
            OutputBucketName=S3_BUCKET_NAME,
            OutputKey=output_file_key,
            Settings={
                "ShowSpeakerLabels": True,
                "MaxSpeakerLabels": 3
            }
        )
    except Exception as e:
        print(f"Failed to start transcription: {e}")
        return

    # Wait for transcription job to complete
    while True:
        response = transcribe_client.get_transcription_job(TranscriptionJobName=job_name)
        status = response["TranscriptionJob"]["TranscriptionJobStatus"]
        if status in ["COMPLETED", "FAILED"]:
            break
        print("Waiting for transcription to complete...")
        time.sleep(10)

    # Get and print the transcript URL if successful
    if status == "COMPLETED":
        transcript_uri = response["TranscriptionJob"]["Transcript"]["TranscriptFileUri"]
        print(f"Transcription completed. Download transcript: {transcript_uri}")
    else:
        print("Transcription job failed.")
