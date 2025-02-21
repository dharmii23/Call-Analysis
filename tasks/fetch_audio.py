import boto3
import requests
from dotenv import load_dotenv
import os
from datetime import datetime

# Load environment variables
load_dotenv()

AWS_REGION = "ap-south-1"
s3_client = boto3.client("s3", region_name=AWS_REGION)
BUCKET_NAME = "my-bucket"  # Replace with your actual bucket name

def generate_file_key():
    """Generates a unique S3 file key using a timestamp."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return f"call_recordings/recording_{timestamp}.mp3"

def fetch_audio_from_kaleyra():
    """Fetches call recording from Kaleyra API and uploads it to S3."""
    from utils.s3_helper import upload_to_s3  # ✅ Delayed import to avoid circular import

    kaleyra_url = os.getenv("KALEYRA_API_URL")
    kaleyra_key = os.getenv("KALEYRA_API_KEY")

    if not kaleyra_key:
        raise ValueError("KALEYRA_API_KEY is missing! Check your .env file.")

    # Fetch call recording from Kaleyra API
    response = requests.get(kaleyra_url, headers={"Authorization": f"Bearer {kaleyra_key}"})
    
    if response.status_code != 200:
        raise Exception(f"Failed to fetch recording: {response.status_code}, {response.text}")

    audio_data = response.content

    # Generate unique filename and upload to S3
    file_key = generate_file_key()
    upload_to_s3(audio_data, BUCKET_NAME, file_key)

    print(f"Audio file stored in S3: {get_s3_file_url(BUCKET_NAME, file_key)}")
