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

def upload_to_s3(file_data, bucket_name, file_key):
    """Uploads a file to S3"""
    try:
        s3_client.put_object(Bucket=bucket_name, Key=file_key, Body=file_data)
        print(f" File uploaded to S3: s3://{bucket_name}/{file_key}")
        return True
    except Exception as e:
        print(f" Failed to upload file: {e}")
        return False

def fetch_transcript_from_s3(bucket_name, file_key=None):
    """Fetches transcript files dynamically from S3"""
    if file_key:
        obj = s3_client.get_object(Bucket=bucket_name, Key=file_key)
        return obj["Body"].read().decode("utf-8")
    
    response = s3_client.list_objects_v2(Bucket=bucket_name, Prefix="my-output-files/")
    if "Contents" not in response:
        return []

    return [obj["Key"] for obj in response["Contents"] if obj["Key"].endswith(".txt")]

def get_s3_file_url(bucket_name, file_key):
    """Returns the public URL of an S3 file"""
    return f"https://{bucket_name}.s3.amazonaws.com/{file_key}"

def list_s3_files(bucket_name, prefix=""):
    """Lists files in an S3 bucket under a given prefix"""
    try:
        response = s3_client.list_objects_v2(Bucket=bucket_name, Prefix=prefix)
        if "Contents" in response:
            return [obj["Key"] for obj in response["Contents"]]
        return []
    except Exception as e:
        print(f"Error listing files in S3: {e}")
        return []
