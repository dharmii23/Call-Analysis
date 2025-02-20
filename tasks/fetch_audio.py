import boto3
import requests
from utils.s3_helper import upload_to_s3
from dotenv import load_dotenv
import os

def fetch_audio_from_kaleyra():
    kaleyra_url = os.getenv("KALEYRA_API_URL")
    kaleyra_key = os.getenv("KALEYRA_API_KEY")
    if not kaleyra_key:
        raise ValueError("KALEYRA_API_KEY is missing! Check your .env file.")
    # Fetch call recording from Kaleyra API
    response = requests.get("https://api-voice.kaleyra.com/v1/")
    audio_data = response.content
    
    # Upload to S3
    upload_to_s3(audio_data, "my-bucket", "call_recording.mp3")
    print("Audio file stored in S3")
