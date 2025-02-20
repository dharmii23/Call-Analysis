import boto3

AWS_REGION = "ap-south-1"
s3_client = boto3.client("s3", region_name=AWS_REGION)

def upload_to_s3(file_data, bucket_name, file_key):
    """Uploads a file to S3"""
    try:
        s3_client.put_object(Bucket=bucket_name, Key=file_key, Body=file_data)
        print(f"✅ File uploaded to S3: s3://{bucket_name}/{file_key}")
        return True
    except Exception as e:
        print(f"❌ Failed to upload file: {e}")
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
            return response["Contents"]
        return []
    except Exception as e:
        print(f"❌ Error listing files in S3: {e}")
        return []
