import boto3
import os
import json
from langchain_community.chat_models import ChatOpenAI
from langchain_openai import ChatOpenAI
from langchain.output_parsers import StructuredOutputParser, ResponseSchema
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from datetime import datetime, timedelta

# Load environment variables
load_dotenv()

# AWS Config
AWS_REGION = "ap-south-1"
BUCKET_NAME = os.getenv("BUCKET_NAME")
TRANSCRIPTS_FOLDER = "kaleyra_report/transcriptions"

# Function to get from_date and to_date
def start_and_end_date():
    from_date = (datetime.utcnow() - timedelta(days=7)).strftime("%Y-%m-%d")
    to_date = (datetime.utcnow() - timedelta(days=1)).strftime("%Y-%m-%d")
    return {'from_date': from_date, 'to_date': to_date}

# AWS Clients
s3_client = boto3.client("s3", region_name=AWS_REGION)

# OpenAI Chat Model
llm = ChatOpenAI(model_name="gpt-3.5-turbo", openai_api_key=os.getenv("OPENAI_API_KEY"))

# Define structured response schema
response_schemas = [
    ResponseSchema(name="call_type", description="Categorization of the call (Billing Issue, Technical Support, etc.)"),
    ResponseSchema(name="sentiment", description="Overall sentiment (Positive, Neutral, Negative)"),
    ResponseSchema(name="emotion_detection", description="Detected emotions (frustration, anger, happiness, calmness)"),
    ResponseSchema(name="sentiment_shifts", description="Analysis of how customer sentiment changed throughout the call"),
    ResponseSchema(name="conversation_flow", description="Evaluation of logical progression, clarity, and coherence"),
    ResponseSchema(name="speech_to_silence_ratio", description="Ratio of spoken words to silence periods, highlighting engagement balance"),
    ResponseSchema(name="interruption_rate", description="Frequency and instances where the agent or customer interrupted the other"),
    ResponseSchema(name="question_to_statement_ratio", description="Ratio of questions asked vs. statements made by the agent"),
    ResponseSchema(name="hold_time_analysis", description="Total hold time during the call"),
    ResponseSchema(name="customer_effort_score", description="Evaluation of how easy or difficult it was for the customer to resolve their issue"),
    ResponseSchema(name="issue_detected", description="Key issues mentioned by the customer"),
    ResponseSchema(name="compliance_adherence", description="Assessment of whether the agent followed guidelines and script"),
    ResponseSchema(name="resolution_status", description="Whether the customer's issue was resolved within the call"),
]

output_parser = StructuredOutputParser.from_response_schemas(response_schemas)
format_instructions = output_parser.get_format_instructions()

call_analysis_prompt = PromptTemplate(
    template="""
    You are an AI-powered call analyst specializing in evaluating customer service interactions. 
    Your task is to analyze the call transcript below and extract detailed insights based on the following key performance indicators:
    
    **Transcript:**
    {transcript}
    
    {format_instructions}
    """,
    input_variables=["transcript"],
    partial_variables={"format_instructions": format_instructions},
)

LOCAL_ANALYSIS_FILE = "analysis_results.txt"

def extract_transcript_with_speakers(transcript_data):
    results = transcript_data.get("results", {})
    items = results.get("items", [])
    speaker_segments = results.get("speaker_labels", {}).get("segments", [])

    if not speaker_segments:
        print("Warning: No speaker labels found, assigning default Speaker 1.")
        return "\n".join(["Speaker 1: " + seg["transcript"] for seg in results.get("audio_segments", [])])

    # Mapping timestamps to speaker labels
    speaker_dict = {}
    for segment in speaker_segments:
        speaker_label = segment["speaker_label"]
        for item in segment.get("items", []):
            speaker_dict[item["start_time"]] = speaker_label

    formatted_transcript = []
    current_speaker = None
    current_sentence = []

    for item in items:
        if item["type"] == "pronunciation":
            word = item["alternatives"][0]["content"]
            start_time = item["start_time"]
            
            # Get the speaker, defaulting to Speaker 1 if unknown
            speaker = speaker_dict.get(start_time, "Speaker_1")
            
            # If speaker changes, start a new line
            if current_speaker is None or current_speaker != speaker:
                if current_sentence:
                    formatted_transcript.append(f"{current_speaker}: {' '.join(current_sentence)}")
                    current_sentence = []
                current_speaker = speaker  # Update speaker

            current_sentence.append(word)

        elif item["type"] == "punctuation":
            current_sentence.append(item["alternatives"][0]["content"])  # Add punctuation

    # Append the last sentence
    if current_sentence:
        formatted_transcript.append(f"{current_speaker}: {' '.join(current_sentence)}")

    return "\n".join(formatted_transcript)


def fetch_transcript_texts():
    """Fetch only the transcript text with timestamps and speakers from each JSON file in S3."""
    date = start_and_end_date()
    transcript_prefix = f"{TRANSCRIPTS_FOLDER}/{date['from_date']}-{date['to_date']}"
    response = s3_client.list_objects_v2(Bucket=BUCKET_NAME, Prefix=transcript_prefix)
    transcript_texts = {}

    if "Contents" not in response:
        return transcript_texts  # Return empty if no files

    for obj in response["Contents"]:
        file_key = obj["Key"]

        if not file_key.endswith(".json"):
            continue  # Skip non-JSON files

        try:
            file_obj = s3_client.get_object(Bucket=BUCKET_NAME, Key=file_key)
            file_content = file_obj["Body"].read().decode("utf-8")
            transcript_data = json.loads(file_content)

            formatted_transcript = extract_transcript_with_speakers(transcript_data)
            print(formatted_transcript)  # Debugging

            if formatted_transcript:
                transcript_texts[file_key] = formatted_transcript

        except (json.JSONDecodeError, Exception) as e:
            print(f"Error processing {file_key}: {e}")
            continue  # Skip corrupted files

    return transcript_texts


def analyze_and_store():
    transcripts = fetch_transcript_texts()
    if not transcripts:
        return  # No transcripts available
    
    all_analysis_content = ""
    date = start_and_end_date()
    analysis_filename = f"kaleyra_report/analysis_files/{date['from_date']}-{date['to_date']}.txt"
    s3_analysis_path = f"keywords-bucket/{analysis_filename}"
    
    for file_key, transcript_text in transcripts.items():
        print(file_key)
        prompt = call_analysis_prompt.format(transcript=transcript_text)
        response = llm.invoke(prompt)
        structured_response = output_parser.parse(response.content)

        analysis_text = f"""
        ==============================
        File: {file_key}
        ==============================
        Transcript:
        {transcript_text}
        
        Analysis:
        --------------------------------
        Call Type: {structured_response.get("call_type", "N/A")}
        Sentiment: {structured_response.get("sentiment", "N/A")}
        Detected Emotions: {structured_response.get("emotion_detection", "N/A")}
        Sentiment Shifts: {structured_response.get("sentiment_shifts", "N/A")}
        Speech-to-Silence Ratio: {structured_response.get("speech_to_silence_ratio", "N/A")}
        Interruption Rate: {structured_response.get("interruption_rate", "N/A")}
        Hold Time Analysis: {structured_response.get("hold_time_analysis", "N/A")}
        Customer Effort Score: {structured_response.get("customer_effort_score", "N/A")}
        Compliance Adherence: {structured_response.get("compliance_adherence", "N/A")}
        Resolution Status: {structured_response.get("resolution_status", "N/A")}
        Detected Issue: {structured_response.get("issue_detected", "N/A")}
        --------------------------------
        """
        all_analysis_content += analysis_text.strip() + "\n\n"

    with open(LOCAL_ANALYSIS_FILE, "w", encoding="utf-8") as file:
        file.write(all_analysis_content)

    try:
        s3_client.upload_file(LOCAL_ANALYSIS_FILE, "keywords-bucket", analysis_filename)
        print(f"Successfully uploaded analysis to S3: s3://{s3_analysis_path}")
    except Exception as e:
        print(f"Error uploading file to S3: {e}")

if __name__ == "__main__":
    analyze_and_store()
