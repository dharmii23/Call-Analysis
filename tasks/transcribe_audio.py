import boto3
import os
import json
import time
import requests
from dotenv import load_dotenv
from datetime import datetime, timedelta

# Load environment variablesa
load_dotenv()

# AWS Config
AWS_REGION = "ap-south-1"
BUCKET_NAME = os.getenv("BUCKET_NAME")
TRANSCRIPTS_FOLDER = "kaleyra_report/transcriptions"

# AWS Clients
s3_client = boto3.client("s3", region_name=AWS_REGION)
transcribe_client = boto3.client("transcribe", region_name=AWS_REGION)

def start_and_end_date():
    from_date = (datetime.utcnow() - timedelta(days=7)).strftime("%Y-%m-%d")
    to_date = (datetime.utcnow() - timedelta(days=1)).strftime("%Y-%m-%d")
    return {'from_date': from_date, 'to_date': to_date}

def transcribe_audio_with_aws(audio_s3_uri, job_name):
    """Starts AWS Transcribe job with speaker identification and waits for completion."""
    try:
        date = start_and_end_date()
        
        # Start Transcription with Speaker Labels
        transcribe_client.start_transcription_job(
            TranscriptionJobName=job_name,
            Media={"MediaFileUri": audio_s3_uri},
            MediaFormat="mp3",
            LanguageCode="en-US",
            OutputBucketName=BUCKET_NAME,
            OutputKey=f"{TRANSCRIPTS_FOLDER}/{date['from_date']}-{date['to_date']}/{job_name}.json",
            Settings={
                "ShowSpeakerLabels": True,   # Enable Speaker Labeling
                "MaxSpeakerLabels": 2,       # Maximum 2 speakers
                "ChannelIdentification": False  # Important: Set to False
            }
        )

        while True:
            status = transcribe_client.get_transcription_job(TranscriptionJobName=job_name)
            job_status = status["TranscriptionJob"]["TranscriptionJobStatus"]

            if job_status in ["COMPLETED", "FAILED"]:
                break
            
            print(f"Waiting for transcription... (Status: {job_status})")
            time.sleep(5)

        if job_status == "FAILED":
            print(f"Transcription job {job_name} failed: {status['TranscriptionJob']['FailureReason']}")
            return None

        transcript_url = status["TranscriptionJob"]["Transcript"]["TranscriptFileUri"]

        if not transcript_url:
            print(f"Error: Transcription job {job_name} has no transcript URL.")
            return None

        response = requests.get(transcript_url)
        if response.status_code != 200:
            print(f"Error fetching transcript from {transcript_url}, HTTP Status: {response.status_code}")
            return None
        
        return response.json()

    except Exception as e:
        print(f"Error in AWS Transcribe: {e}")
    
    return None

def process_audio_files():
    """ Processes audio files and saves transcriptions in a structured folder with date-based naming. """
    from_and_to_date = start_and_end_date()
    date_folder = f"transcribe_{from_and_to_date['from_date']}-{from_and_to_date['to_date']}"
    audio_prefix = f"kaleyra_report/call_recordings/{from_and_to_date['from_date']}_To_{from_and_to_date['to_date']}/"
    transcript_prefix = f"{TRANSCRIPTS_FOLDER}/{date_folder}/"

    response = s3_client.list_objects_v2(Bucket=BUCKET_NAME, Prefix=audio_prefix)
    if "Contents" not in response:
        print("No audio files found.")
        return

    for obj in response["Contents"]:
        audio_s3_key = obj["Key"]
        unique_id = os.path.basename(audio_s3_key).replace(".mp3", "")

        print(f"Processing: {audio_s3_key} (ID: {unique_id})")
        audio_s3_uri = f"s3://{BUCKET_NAME}/{audio_s3_key}"

        # Ensure unique job name by appending a timestamp
        timestamp = datetime.utcnow().strftime("%Y%m%d%H%M%S")
        job_name = f"{unique_id}-{timestamp}"

        transcription_data = transcribe_audio_with_aws(audio_s3_uri, job_name)

        if transcription_data:
            transcript_s3_key = f"{transcript_prefix}{unique_id}.json"
            s3_client.put_object(Bucket=BUCKET_NAME, Key=transcript_s3_key, Body=json.dumps(transcription_data), ContentType="application/json")
            print(f"Saved to: s3://{BUCKET_NAME}/{transcript_s3_key}")

if __name__ == "__main__":
    process_audio_files()
