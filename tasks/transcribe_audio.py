import boto3
import os
import json
import time
from dotenv import load_dotenv
from datetime import datetime, timedelta
import pytz

# Load environment variables
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
    """ Starts AWS Transcribe job and waits for completion """
    try:
        transcribe_client.start_transcription_job(
            TranscriptionJobName=job_name,
            Media={"MediaFileUri": audio_s3_uri},
            MediaFormat="mp3",
            LanguageCode="en-US",
            OutputBucketName=BUCKET_NAME  # AWS Transcribe saves results here
        )

        while True:
            status = transcribe_client.get_transcription_job(TranscriptionJobName=job_name)
            if status["TranscriptionJob"]["TranscriptionJobStatus"] in ["COMPLETED", "FAILED"]:
                break
            print("Waiting for transcription...")
            time.sleep(5)

        if status["TranscriptionJob"]["TranscriptionJobStatus"] == "COMPLETED":
            transcript_url = status["TranscriptionJob"]["Transcript"]["TranscriptFileUri"]
            return requests.get(transcript_url).json()

    except boto3.exceptions.Boto3Error as e:
        print(f"Error in AWS Transcribe: {e}")
    
    return None

def process_audio_files():
    """ Processes audio files and saves transcriptions in a structured folder. """
    from_and_to_date = start_and_end_date()
    date_folder = f"{from_and_to_date['from_date']}_To_{from_and_to_date['to_date']}"
    audio_prefix = f"kaleyra_report/call_recordings/{date_folder}/"
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

        transcription_data = transcribe_audio_with_aws(audio_s3_uri, f"transcription-{unique_id}")

        if transcription_data:
            transcript_s3_key = f"{transcript_prefix}{unique_id}.json"
            s3_client.put_object(Bucket=BUCKET_NAME, Key=transcript_s3_key, Body=json.dumps(transcription_data), ContentType="application/json")
            print(f"Saved to: s3://{BUCKET_NAME}/{transcript_s3_key}")

if __name__ == "__main__":
    process_audio_files()
