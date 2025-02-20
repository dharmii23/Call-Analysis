import json
from utils.s3_helper import fetch_transcript_from_s3, upload_to_s3
from dotenv import load_dotenv
import os
load_dotenv()
from langchain_openai import ChatOpenAI  # Updated import
from langchain.schema import HumanMessage
from prompts.prompts import call_analysis_prompt, output_parser

# Buckets
TRANSCRIPTS_BUCKET = "keywords-bucket"
ANALYSIS_BUCKET = "call-analysis"

# Initialize OpenAI model
llm = ChatOpenAI(model_name="gpt-4", temperature=0)

def analyze_transcript():
    openai_key = os.getenv("OPENAI_API_KEY")
    
    if not openai_key:
        raise ValueError("OPENAI_API_KEY is missing! Check your .env file.")

    """Fetches transcript from S3, analyzes it with GPT, and saves results to another S3 bucket."""
    
    s3_files = fetch_transcript_from_s3(TRANSCRIPTS_BUCKET)  # Get list of all transcript files

    for file_key in s3_files:
        call_id = file_key.replace(".txt", "")  # Extract call ID from filename
        print(f"🔍 Processing: {file_key}")

        transcript_text = fetch_transcript_from_s3(TRANSCRIPTS_BUCKET, file_key)

        if not transcript_text:
            print(f"❌ No transcript found for {file_key}")
            continue

        # Generate analysis using GPT
        formatted_prompt = call_analysis_prompt.format(transcript=transcript_text)
        messages = [HumanMessage(content=formatted_prompt)]
        response = llm(messages)
        call_analysis_result = output_parser.parse(response.content)

        # Convert analysis result to string
        analysis_text = json.dumps(call_analysis_result, indent=4)

        # Save the analysis as a .txt file in S3 bucket
        analysis_file_key = f"{call_id}_analysis.txt"
        upload_to_s3(analysis_text.encode("utf-8"), ANALYSIS_BUCKET, analysis_file_key)

        print(f"✅ Analysis saved: s3://{ANALYSIS_BUCKET}/{analysis_file_key}")
