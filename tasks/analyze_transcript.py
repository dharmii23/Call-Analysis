import boto3
import os
import json
from langchain_openai import ChatOpenAI
from langchain.output_parsers import StructuredOutputParser, ResponseSchema
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from datetime import datetime, timedelta
from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, PageBreak, Table, TableStyle
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib import colors

# Load environment variables
load_dotenv()

# AWS Config
AWS_REGION = "ap-south-1"
BUCKET_NAME = os.getenv("BUCKET_NAME")
TRANSCRIPTS_FOLDER = "kaleyra_report/transcriptions"

# ✅ Function to get date range
def start_and_end_date():
    from_date = (datetime.utcnow() - timedelta(days=7)).strftime("%Y-%m-%d")
    to_date = (datetime.utcnow() - timedelta(days=1)).strftime("%Y-%m-%d")
    return {'from_date': from_date, 'to_date': to_date}

# ✅ Function to extract only dialogues from JSON
def extract_dialogues_only(transcript_data):
    results = transcript_data.get("results", {})
    audio_segments = results.get("audio_segments", [])

    if not audio_segments:
        return "No transcript available."

    dialogues = []
    for segment in audio_segments:
        speaker = segment.get("speaker_label", "Unknown Speaker")
        text = segment.get("transcript", "").strip()
        if text:
            dialogues.append(f"{speaker}: {text}")

    return "\n".join(dialogues)


# ✅ Function to trim transcript (if it's too long)
def trim_transcript(text, max_words=5000):
    words = text.split()
    if len(words) > max_words:
        print(f"Transcript too long ({len(words)} words), trimming...")
        return " ".join(words[:max_words]) + "... [Transcript shortened]"
    return text






# ✅ Function to save PDF
def save_as_pdf(content, filename):
    doc = SimpleDocTemplate(filename, pagesize=letter, leftMargin=72, rightMargin=72, topMargin=72, bottomMargin=72)
    styles = getSampleStyleSheet()
    elements = []

    # Loop through all sections in content to add them to the PDF
    for section in content:
    # ✅ Skip sections without a "title" key
        if "title" not in section:
            print(f"Skipping section without title: {section}")  # Debugging step
            continue

        elements.append(Paragraph(section["title"], styles["Heading2"]))
        elements.append(Spacer(1, 12))

        if section["title"].startswith("Transcript"):
            elements.append(Paragraph(section["text"].replace("\n", "<br/>"), styles["BodyText"]))
            elements.append(PageBreak())

        elif section.get("text"):
            elements.append(Paragraph(section["text"], styles["BodyText"]))
            elements.append(Spacer(1, 20))

        elif section.get("tables"):
            for table_data in section["tables"]:
                formatted_table_data = [[Paragraph(str(cell), styles["BodyText"]) for cell in row] for row in table_data]
                table = Table(formatted_table_data)
                table.setStyle(TableStyle([
                ('GRID', (0, 0), (-1, -1), 1, colors.black),
                ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
                ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
                ('BACKGROUND', (0, 0), (-1, 0), colors.grey),
                ('FONTNAME', (0, 0), (-1, -1), 'Helvetica'),
                ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
                ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
            ]))

            elements.append(table)
            elements.append(Spacer(1, 20))

    doc.build(elements)



# ✅ Function to generate weekly content
def generate_weekly_content(transcripts, date_range):
    weekly_data = {
        "total_calls": len(transcripts),
        "from_date": date_range['from_date'],
        "to_date": date_range['to_date']
    }

    # Calculate average sentiment and call handling time from transcripts here
    # (You can define your logic for these metrics)

    # Create the prompt for GPT to generate the weekly summary
    weekly_summary_prompt = f"""
Summarize the overall customer service performance for the week based on the following data:

- Describe customer sentiment trends (positive, neutral, or negative) and provide reasons.
- Highlight common complaints from customers.
- Evaluate agent performance and suggest improvements.

**Write a professional, detailed summary in paragraph form. Do not return structured JSON.**
"""

    # Ask GPT to generate the weekly summary paragraph
    weekly_summary_response = llm.invoke(weekly_summary_prompt)
    weekly_summary_paragraph = weekly_summary_response.content.strip()
    print(weekly_summary_response)
    return weekly_summary_paragraph, weekly_data

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
    and generate structured reports.

    *Transcript:*  
    {transcript}

    *Metrics & Analysis:*  
    {format_instructions}
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
            dialogues = extract_dialogues_only(transcript_data)
            transcript_texts[file_key] = dialogues
        except json.JSONDecodeError:
            continue

    return transcript_texts

# ✅ Analyze and store results
def analyze_and_store():
    transcripts = fetch_transcripts()
    if not transcripts:
        print("No transcripts found.")
        return

    date_range = start_and_end_date()  # Get the date range
    analysis_filename = f"kaleyra_report/analysis_files/{date_range['from_date']}-{date_range['to_date']}.pdf"
    s3_analysis_path = f"keywords-bucket/{analysis_filename}"

    content = []

    # Add weekly analysis content with dates
    weekly_summary_paragraph, weekly_content = generate_weekly_content(transcripts, date_range)
    # print("Content1:", weekly_summary_paragraph)
    # print("Contnent2:", weekly_content)

    print(f"Weekly Summary Paragraph: {weekly_summary_paragraph}")

    content.append({
    "title": f"Weekly Summary ({weekly_content['from_date']} to {weekly_content['to_date']})",
})


    content.append({
    "title": "Summary of the Week",
    "text": weekly_summary_paragraph
})
    
    content.append({
    "tables": [[
        ["Metric", "Value"],
        ["Total Calls", weekly_content["total_calls"]],
    ]]
})

    # Add weekly metrics table
    # Add weekly summary at the beginning of the PDF

# ✅ Add the weekly summary paragraph right after the table



    for file_key, transcript_text in transcripts.items():
        print(f"Processing: {file_key}")

        trimmed_transcript = trim_transcript(transcript_text)  # Trim if too long
        response = llm.invoke(call_analysis_prompt.format(transcript=trimmed_transcript))
        # print(response.content)
        structured_response = output_parser.parse(response.content)

        # ✅ Add cleaned transcript
        content.append({
            "title": f"Transcript: {file_key}",
            "text": trimmed_transcript
        })

        # ✅ Add agent performance metrics
        content.append({
            "title": "Agent Performance Metrics",
            "tables": [[
                ["Metric", "Value"],
                ["Productivity", structured_response.get("productivity", "N/A")],
                ["Quality & Experience", structured_response.get("quality_experience", "N/A")],
                ["Compliance", structured_response.get("compliance", "N/A")],
                ["Call Handling", structured_response.get("call_handling", "N/A")]
            ]]
        })

        # ✅ Add customer sentiment analysis metrics
        content.append({
            "title": "Customer Sentiment Analysis Metrics",
            "tables": [[
                ["Metric", "Value"],
                ["Overall Sentiment Score", structured_response.get("customer_sentiment", "N/A")],
                ["Emotion Detection", structured_response.get("emotion_detection", "N/A")],
                ["Sentiment Shift", structured_response.get("sentiment_shift", "N/A")],
                ["Escalation Risk", structured_response.get("escalation_risk", "N/A")]
            ]]
        })

    # Save the PDF and upload to S3
    save_as_pdf(content, "analysis_results.pdf")
    s3_client.upload_file("analysis_results.pdf", "keywords-bucket", analysis_filename)
    print(f"Successfully uploaded analysis to S3: s3://{s3_analysis_path}")
if __name__ == "__main__":
    analyze_and_store()
