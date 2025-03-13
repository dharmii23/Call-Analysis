import boto3
import os
import json
from langchain_openai import ChatOpenAI
from langchain.output_parsers import StructuredOutputParser, ResponseSchema
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from datetime import datetime, timedelta
from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Table, TableStyle, Paragraph, Spacer, PageBreak
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib import colors


# Load environment variables
load_dotenv()

# AWS Config
AWS_REGION = "ap-south-1"
BUCKET_NAME = os.getenv("BUCKET_NAME")
TRANSCRIPTS_FOLDER = "kaleyra_report/transcriptions"

# Function to get date range
def start_and_end_date():
    from_date = (datetime.utcnow() - timedelta(days=7)).strftime("%Y-%m-%d")
    to_date = (datetime.utcnow() - timedelta(days=1)).strftime("%Y-%m-%d")
    return {'from_date': from_date, 'to_date': to_date}


# ✅ Function to extract only dialogues from JSON
def extract_dialogues_only(transcript_data):
    """Extracts speaker dialogues in 'Speaker: Dialogue' format."""
    results = transcript_data.get("results", {})
    audio_segments = results.get("audio_segments", [])

    if not audio_segments:
        print("Warning: No dialogues found, returning empty transcript.")
        return "No transcript available."

    dialogues = []
    for segment in audio_segments:
        speaker = segment.get("speaker_label", "Unknown Speaker")
        text = segment.get("transcript", "").strip()
        if text:  # Ignore empty segments
            dialogues.append(f"{speaker}: {text}")

    return "\n".join(dialogues)


# ✅ Function to trim transcript (if it's too long)
def trim_transcript(text, max_words=5000):
    words = text.split()
    if len(words) > max_words:
        print(f"Transcript too long ({len(words)} words), trimming...")
        return " ".join(words[:max_words]) + "... [Transcript shortened]"
    return text


# ✅ Function to save analysis as a PDF
def save_as_pdf(content, filename):
    doc = SimpleDocTemplate(filename, pagesize=letter, leftMargin=72, rightMargin=72, topMargin=72, bottomMargin=72)
    styles = getSampleStyleSheet()
    elements = []

    for section in content:
        elements.append(Paragraph(section["title"], styles["Heading2"]))
        elements.append(Spacer(1, 12))

        if section["title"].startswith("Transcript"):
            elements.append(Paragraph(section["text"].replace("\n", "<br/>"), styles["BodyText"]))
            elements.append(PageBreak())

        else:
            if "tables" in section:
                for table_data in section["tables"]:
                    formatted_table_data = [[Paragraph(str(cell), styles["BodyText"]) for cell in row] for row in table_data]
                    table = Table(formatted_table_data)
                    table.setStyle(TableStyle([('GRID', (0, 0), (-1, -1), 1, colors.black)]))
                    elements.append(table)
                    elements.append(Spacer(1, 20))

    doc.build(elements)


# ✅ AWS S3 Client
s3_client = boto3.client("s3", region_name=AWS_REGION)

# ✅ OpenAI Chat Model
llm = ChatOpenAI(model_name="gpt-3.5-turbo", openai_api_key=os.getenv("OPENAI_API_KEY"))

# ✅ Define structured response schema
response_schemas = [
    ResponseSchema(name="productivity", description="Productivity metrics."),
    ResponseSchema(name="quality_experience", description="Quality & Experience metrics."),
    ResponseSchema(name="compliance", description="Compliance metrics."),
    ResponseSchema(name="call_handling", description="Call Handling metrics."),
    ResponseSchema(name="customer_sentiment", description="Overall Sentiment Score."),
    ResponseSchema(name="emotion_detection", description="Detected emotions."),
    ResponseSchema(name="sentiment_shift", description="Sentiment change across call."),
    ResponseSchema(name="escalation_risk", description="Likelihood of escalation."),
    ResponseSchema(name="customer_effort", description="Customer Effort Score."),
    ResponseSchema(name="top_complaints", description="List of top customer complaints."),
    ResponseSchema(name="agent_feedback", description="Personalized agent feedback."),
]

output_parser = StructuredOutputParser.from_response_schemas(response_schemas)
format_instructions = output_parser.get_format_instructions()

# ✅ Call Analysis Prompt
call_analysis_prompt = PromptTemplate(
    template="""
    You are an AI call analyst. Analyze the following customer service call transcript 
    and generate structured reports. Also generate a structured Weekly Call Performance Summary.

    *Transcript:*  
    {transcript}

    *Metrics & Analysis:*  
    {format_instructions}

     *Weekly Call Reports:*  
    {transcripts}

    *Weekly Summary:*  
    - Total number of calls analyzed  
    - Overall sentiment distribution (positive, neutral, negative)  
    - Most common complaints and their frequency  
    - Key customer pain points  
    - Any patterns or trends observed in customer interactions  
    """,
    input_variables=["transcript"],
    partial_variables={"format_instructions": format_instructions},
)


# ✅ Fetch transcripts from S3
def fetch_transcripts():
    date = start_and_end_date()
    transcript_prefix = f"{TRANSCRIPTS_FOLDER}/{date['from_date']}-{date['to_date']}"
    response = s3_client.list_objects_v2(Bucket=BUCKET_NAME, Prefix=transcript_prefix)
    transcript_texts = {}

    if "Contents" not in response:
        return transcript_texts

    for obj in response["Contents"]:
        file_key = obj["Key"]
        if not file_key.endswith(".json"):
            continue

        file_obj = s3_client.get_object(Bucket=BUCKET_NAME, Key=file_key)
        file_content = file_obj["Body"].read().decode("utf-8")

        try:
            transcript_data = json.loads(file_content)
            dialogues = extract_dialogues_only(transcript_data)  # Extract cleaned dialogues
            transcript_texts[file_key] = dialogues
        except json.JSONDecodeError as e:
            print(f"Error parsing {file_key}: {e}")
            continue

    return transcript_texts


# ✅ Analyze and store results
def analyze_and_store():
    transcripts = fetch_transcripts()
    if not transcripts:
        print("No transcripts found.")
        return

    date = start_and_end_date()
    analysis_filename = f"kaleyra_report/analysis_files/{date['from_date']}-{date['to_date']}.pdf"
    s3_analysis_path = f"keywords-bucket/{analysis_filename}"

    content = []
    weekly_summary_data=[]

    for file_key, transcript_text in transcripts.items():
        print(f"Processing: {file_key}")

        trimmed_transcript = trim_transcript(transcript_text)  # Trim if too long
        response = llm.invoke(call_analysis_prompt.format(transcript=trimmed_transcript))
        structured_response = output_parser.parse(response.content)

        weekly_summary_data.append({
            "transcript_id": file_key,
            "sentiment": structured_response.get("customer_sentiment", "Neutral"),
            "top_complaints": structured_response.get("top_complaints", [])
        })


        # ✅ Add cleaned transcript
        content.append({
            "title": f"Transcript: {file_key}",
            "text": trimmed_transcript
        })

        # ✅ Add performance metrics
        content.append({
            "title": "Agent Performance Metrics",
            "tables": [[["Metric", "Value"],
                        ["Productivity", structured_response.get("productivity", "N/A")],
                        ["Quality & Experience", structured_response.get("quality_experience", "N/A")],
                        ["Compliance", structured_response.get("compliance", "N/A")],
                        ["Call Handling", structured_response.get("call_handling", "N/A")]]]
        })

        # ✅ Add sentiment analysis metrics
        content.append({
            "title": "Customer Sentiment Analysis Metrics",
            "tables": [[["Metric", "Value"],
                        ["Overall Sentiment Score", structured_response.get("customer_sentiment", "N/A")],
                        ["Emotion Detection", structured_response.get("emotion_detection", "N/A")],
                        ["Sentiment Shift", structured_response.get("sentiment_shift", "N/A")],
                        ["Escalation Risk", structured_response.get("escalation_risk", "N/A")]]]
        })

         # ✅ Generate Weekly Summary Using GPT
    print("🔍 DEBUG: Transcript before GPT call:")
    print(weekly_summary_data)

    if weekly_summary_data:  # ✅ Only run GPT if data exists
        weekly_summary_prompt = f"""
    Given the following analyzed call data, summarize the overall call performance for the week:

    {json.dumps(weekly_summary_data, indent=2)}

    Provide a concise paragraph about the trends in customer sentiment, common issues, and overall agent performance.
    """
        weekly_summary_response = llm.invoke(weekly_summary_prompt).content
        print("DEBUG: Weekly Summary Before Report Generation:", weekly_summary_response)

        content.append({"title": "📌 Weekly Call Performance Summary", "text": weekly_summary_response})
    else:
        print("⚠️ Warning: No data found for Weekly Performance Summary. Skipping GPT call.")


    # ✅ Add GPT-generated final summary to the PDF
    content.append({"title": "📌 Weekly Call Performance Summary", "text": weekly_summary_response})

    save_as_pdf(content, "analysis_results.pdf")
    s3_client.upload_file("analysis_results.pdf", "keywords-bucket", analysis_filename)
    print(f"Successfully uploaded analysis to S3: s3://{s3_analysis_path}")


if __name__ == "__main__":
    analyze_and_store()