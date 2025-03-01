import boto3
import requests
from dotenv import load_dotenv
import os
from datetime import datetime, timedelta, date
import pytz
from typing import Dict
from io import BytesIO
import pandas as pd


load_dotenv()

AWS_REGION = os.getenv("AWS_REGION")
BUCKET_NAME = os.getenv("BUCKET_NAME")
s3_client = boto3.client("s3", region_name=AWS_REGION)

def generate_file_key(from_date: date, to_date: date, unique_id: str) -> str:
    """Generates a unique S3 file key using an unique id."""
    return f"kaleyra_report/call_recordings/{from_date}_To_{to_date}/{unique_id}.mp3"

def start_and_end_date()-> Dict[str, date]:
    utc_zone = pytz.utc
    ist_zone = pytz.timezone("Asia/Kolkata")

    utc_previous = datetime.utcnow() - timedelta(days=7)
    utc_time_previous = utc_zone.localize(utc_previous)
    ist_time_previous = utc_time_previous.astimezone(ist_zone)
    from_date = ist_time_previous.strftime("%Y-%m-%d")

    utc_now = datetime.utcnow() - timedelta(days=1)
    utc_time_now = utc_zone.localize(utc_now)
    ist_time_now = utc_time_now.astimezone(ist_zone)
    to_date = ist_time_now.strftime("%Y-%m-%d")

    return {'from_date': from_date, 'to_date': to_date}

def upload_csv_to_s3(df, from_and_to_date):
    """Uploads the CSV to S3."""
    csv_buffer = BytesIO()
    df.to_csv(csv_buffer, index=False)
    csv_buffer.seek(0)

    csv_file_key = f"kaleyra_report/csv_files/{from_and_to_date.get('from_date')}_To_{from_and_to_date.get('to_date')}.csv"

    s3_client.put_object(Bucket=BUCKET_NAME, Key=csv_file_key, Body=csv_buffer, ContentType='text/csv')
    print(f"CSV uploaded to S3 with key: {csv_file_key}")

def fetch_audio_from_kaleyra_and_upload():
    """Fetches call recording from Kaleyra API and uploads it to S3."""

    kaleyra_url = os.getenv("KALEYRA_API_URL")
    kaleyra_key = os.getenv("KALEYRA_API_KEY")

    if not kaleyra_key:
        raise ValueError("KALEYRA_API_KEY is missing! Check your .env file.")
    
    from_and_to_date = start_and_end_date()

    payload={
        'method': 'dial',
        'format': 'json',
        'fromdate': from_and_to_date.get('from_date'),
        'todate': from_and_to_date.get('to_date'),
        'limit': 604800  # Because max possible call = 24*60*60*7, assume one call minimum took 1 sec.
        }

    headers = {
    'x-api-key': kaleyra_key
    }

    while True:
        response = requests.request("POST", kaleyra_url, headers=headers, data=payload)
        if response.status_code == 200:
            break
    
    if response.status_code != 200:
        raise Exception(f"Failed to fetch recording: {response.status_code}, {response.text}")

    kaleyra_response = response.json()
    save_list = list()
    data_list = kaleyra_response.get('data')

    for element in data_list:
        if element.get('recording'):
            fetch_audio_url = element.get('recording')
            if fetch_audio_url.startswith('//'):
                fetch_audio_url = 'https:' + fetch_audio_url

            fetch_audio = requests.get(fetch_audio_url)

            if fetch_audio.status_code == 200:
                file_name = generate_file_key(from_date=from_and_to_date.get('from_date'), to_date=from_and_to_date.get('to_date'), unique_id=element.get('id'))
                s3_client.put_object(Bucket=BUCKET_NAME, Key=file_name, Body=BytesIO(fetch_audio.content), ContentType='audio/mpeg')
                element["file_name"] = file_name
                save_list.append(element)
                print(f"Storing...file {file_name} successfully")
            else:
                print(f"Failed to fetch recording for id {element.get('id')}")
                continue
    
    df = pd.DataFrame(save_list)
    df.to_csv(f"upload/{from_and_to_date.get('from_date')}_To_{from_and_to_date.get('to_date')}.csv", index=False)
    upload_csv_to_s3(df, from_and_to_date)


if __name__ == "__main__":
    #pass

    fetch_audio_from_kaleyra_and_upload()