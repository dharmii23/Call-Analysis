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
from collections import Counter

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
# ✅ Function to extract dialogues and duration
# ✅ Function to extract dialogues and duration safely
def extract_dialogues_and_duration(transcript_data):
    results = transcript_data.get("results", {})
    audio_segments = results.get("audio_segments", [])

    if not audio_segments:
        return "No transcript available.", 0.0  # Return default if missing

    dialogues = []
    total_duration = 0.0

    for segment in audio_segments:
        speaker = segment.get("speaker_label", "Unknown Speaker")
        text = segment.get("transcript", "").strip()

        # Convert start_time & end_time to float and calculate duration
        try:
            start_time = float(segment.get("start_time", "0.0"))
            end_time = float(segment.get("end_time", "0.0"))
            duration = round(end_time - start_time, 2)
            total_duration += duration  # Sum up total call duration
        except ValueError:
            print(f"⚠ Error converting time values in segment: {segment}")
            continue  # Skip if time values are invalid

        if text:
            dialogues.append(f"{speaker}: {text} (Duration: {duration}s)")

    return "\n".join(dialogues), round(total_duration, 2)




# ✅ Function to trim transcript (if it's too long)
def trim_transcript(text, max_words=5000):
    words = text.split()
    if len(words) > max_words:
        print(f"Transcript too long ({len(words)} words), trimming...")
        return " ".join(words[:max_words]) + "... [Transcript shortened]"
    return text

def analyze_call_with_ai(transcript):
    ai_prompt = f"""
    Analyze the following customer service call and determine its overall sentiment:
    - Options: "Positive", "Neutral", or "Negative"
    - Consider customer tone, agent response, and resolution.

    Also provide conversational metrics:
    - Rate flow smoothness (1-100)
    - Count question vs. statement ratio
    - Identify resolution path efficiency

    Transcript:
    {transcript}

    Provide output in JSON format:
    {{
        "Sentiment": "Positive | Neutral | Negative",
        "Conversation Flow Score": (1-100),
        "Question-to-Statement Ratio": (float),
        "Resolution Path Efficiency": (1-100)
    }}
    """

    response = llm.invoke(ai_prompt)
    
    try:
        return json.loads(response.content.strip())  # Parse response as JSON
    except json.JSONDecodeError:
        print(f"⚠ AI response could not be parsed: {response.content}")
        return None

def save_weekly_pdf(weekly_summary, agent_performance, customer_sentiment, conversation_analysis, filename):
    doc = SimpleDocTemplate(filename, pagesize=letter, leftMargin=72, rightMargin=72, topMargin=72, bottomMargin=72)
    styles = getSampleStyleSheet()
    
    # Define Custom Styles
    title_style = styles["Heading2"]
    body_style = ParagraphStyle(
        "BodyStyle",
        parent=styles["BodyText"],
        leading=16,  # Adjust line spacing for better readability
        spaceAfter=10
    )
    bullet_style = ParagraphStyle(
        "BulletStyle",
        parent=styles["BodyText"],
        bulletIndent=15,
        leftIndent=30,
        leading=14
    )

    elements = []

    # ✅ Weekly Summary
    elements.append(Paragraph("📌 Weekly Call Performance Summary", title_style))
    elements.append(Spacer(1, 12))
    elements.append(Paragraph(weekly_summary.replace("\n", "<br/>"), body_style))
    elements.append(Spacer(1, 20))

    # ✅ Agent Performance Table
    elements.append(Paragraph("🔹 Agent Performance Metrics", title_style))
    agent_table_data = [["Metric", "Value"]] + [[k, str(v)] for k, v in agent_performance.items()]
    table = Table(agent_table_data, colWidths=[250, 150])
    table.setStyle(TableStyle([
        ('GRID', (0, 0), (-1, -1), 1, colors.black),
        ('BACKGROUND', (0, 0), (-1, 0), colors.grey),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.white),
        ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
        ('FONTNAME', (0, 0), (-1, -1), 'Helvetica-Bold'),
        ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
    ]))
    elements.append(table)
    elements.append(Spacer(1, 20))

    # ✅ Customer Sentiment Analysis Table
    elements.append(Paragraph("🔹 Customer Sentiment Analysis", title_style))
    sentiment_table_data = [["Metric", "Value"]] + [[k, str(v)] for k, v in customer_sentiment.items()]
    table = Table(sentiment_table_data, colWidths=[250, 150])
    table.setStyle(TableStyle([
        ('GRID', (0, 0), (-1, -1), 1, colors.black),
        ('BACKGROUND', (0, 0), (-1, 0), colors.grey),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.white),
        ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
        ('FONTNAME', (0, 0), (-1, -1), 'Helvetica-Bold'),
        ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
    ]))
    elements.append(table)
    elements.append(Spacer(1, 20))

    # ✅ Conversation Analysis Table
    elements.append(Paragraph("🔹 Conversation Analysis Metrics", title_style))
    conversation_table_data = [["Metric", "Value"]] + [[k, str(v)] for k, v in conversation_analysis.items()]
    table = Table(conversation_table_data, colWidths=[250, 150])
    table.setStyle(TableStyle([
        ('GRID', (0, 0), (-1, -1), 1, colors.black),
        ('BACKGROUND', (0, 0), (-1, 0), colors.grey),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.white),
        ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
        ('FONTNAME', (0, 0), (-1, -1), 'Helvetica-Bold'),
        ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
    ]))
    elements.append(table)
    elements.append(Spacer(1, 20))

    # ✅ Final Page Break
    elements.append(Spacer(1, 50))

    doc.build(elements)



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

# Helper function to safely extract numeric values
def safe_float(value, default=0):
    try:
        return float(value)  # Convert only if it's a valid number
    except (ValueError, TypeError):
        return default  # Return default if conversion fails

# ✅ Function to generate weekly content
# ✅ Function to generate weekly content
    

from collections import Counter
# ✅ Function to extract sentiment, agent performance, and conversation metrics
def extract_call_metrics(transcript_data):
    results = transcript_data.get("results", {})
    audio_segments = results.get("audio_segments", [])

    if not audio_segments:
        print("⚠ No 'audio_segments' found! Check JSON structure.")
        return {
            "Overall Sentiment Score": 0,
            "Positive": 0,
            "Neutral": 0,
            "Negative": 0,
            "Escalation Risk Score": 0,
            "Frustration Score": 0,
            "Conversation Flow Score": 0,
            "Speech-to-Silence Ratio": 0,
            "Interruption Rate": 0,
            "Hold Time Analysis": 0,
            "Call Length": 0
        }
    total_speech_time = 0.0
    total_silence_time = 0.0
    interruptions = 0
    last_speaker = None
    hold_times = []

    sentiment_counts = Counter()
    total_segments = len(audio_segments)
    frustration_mentions = 0
    escalation_risk = 0
    total_effort_score = 0
    total_silence_duration = 0.0
    total_speech_duration = 0.0
    interruptions = 0
    last_speaker = None

    for segment in audio_segments:
        sentiment = segment.get("sentiment", "Neutral")  # Assume 'Neutral' if sentiment is missing
        sentiment_counts[sentiment] += 1

    positive_count = sentiment_counts.get("Positive", 0)
    neutral_count = sentiment_counts.get("Neutral", 0)
    negative_count = sentiment_counts.get("Negative", 0)

    # Compute the overall sentiment score
    overall_sentiment_score = (positive_count + neutral_count + negative_count) / 3

    for i, segment in enumerate(audio_segments):
        speaker = segment.get("speaker_label", "Unknown Speaker")
        transcript_text = segment.get("transcript", "").lower()
        start_time = float(segment.get("start_time", "0"))
        end_time = float(segment.get("end_time", "0"))
        duration = end_time - start_time

        # Track speech duration per speaker
        total_speech_duration += duration

        # Detect frustration (basic keyword detection)
        if any(word in transcript_text for word in ["frustrated", "angry", "disappointed", "issue"]):
            frustration_mentions += 1

        # Detect escalation (more negative words)
        if any(word in transcript_text for word in ["complain", "supervisor", "escalate", "not happy"]):
            escalation_risk += 1

        # Detect interruptions
        if last_speaker and last_speaker != speaker:
            interruptions += 1

        last_speaker = speaker

        # Estimate customer effort score (longer sentences = higher effort)
        response_length = len(transcript_text.split())
        if response_length > 10:
            total_effort_score += 3
        elif response_length > 5:
            total_effort_score += 2
        else:
            total_effort_score += 1
        
        # ✅ Calculate hold time (silence between responses)
        if i > 0:
            prev_end_time = float(audio_segments[i - 1]["end_time"])
            silence_duration = start_time - prev_end_time
            if silence_duration > 0:
                total_silence_time += silence_duration
                hold_times.append(silence_duration)

        # Calculate silence between segments
        if i > 0:
            prev_end_time = float(audio_segments[i - 1]["end_time"])
            silence_duration = start_time - prev_end_time
            if silence_duration > 0:
                total_silence_duration += silence_duration

    # Normalize values
    total_time = total_speech_duration + total_silence_duration
    speech_silence_ratio = round((total_speech_duration / total_time) * 100, 2) if total_time > 0 else 0
    interruption_rate = round((interruptions / total_segments) * 100, 2) if total_segments > 0 else 0
    hold_time_analysis = round((total_silence_duration / total_segments) * 100, 2) if total_segments > 0 else 0
    total_call_time = total_speech_time + total_silence_time

    return {
        "Overall Sentiment Score":  overall_sentiment_score ,
          # Mock values
        "Customer Effort Score": round(total_effort_score / total_segments, 2) if total_segments else 0,
        "Escalation Risk Score": round((escalation_risk / total_segments) * 100, 2) if total_segments else 0,
        "Frustration Score": round((frustration_mentions / total_segments) * 100, 2) if total_segments else 0,
        "Conversation Flow Score": round(100 - interruption_rate, 2),  # Less interruptions = better flow
        "Speech-to-Silence Ratio": speech_silence_ratio,
        "Interruption Rate": interruption_rate,
        "Hold Time Analysis": hold_time_analysis,
        "Speech-to-Silence Ratio": round((total_speech_time / total_call_time) * 100, 2) if total_call_time else 0,
        "Interruption Rate": round((interruptions / len(audio_segments)) * 100, 2) if len(audio_segments) else 0,
        "Hold Time Analysis": round(sum(hold_times) / len(hold_times), 2) if hold_times else 0,
        "Call Length": round(total_call_time, 2)
    }


def generate_weekly_content(transcripts, date_range):
    total_calls = len(transcripts)
    total_duration = sum(transcript["duration"] for transcript in transcripts.values())
    average_duration = round(total_duration / total_calls, 2) if total_calls else 0

    sentiment_counts = Counter({"Positive": 0, "Neutral": 0, "Negative": 0})

    complaints_list = []

    # ✅ Initialize Sentiment and Conversation Metrics
    customer_sentiment = {

        "Overall Sentiment Score": 0,
        "Positive": 0,
        "Neutral": 0,
        "Negative": 0,
    }

    conversation_analysis = {
        "Conversation Flow Score": 0,
         "Question-to-Statement Ratio": 0,
        "Resolution Path Efficiency": 0,
    }

    # ✅ Process each transcript
    for file_key, transcript_text in transcripts.items():

        dialogues = transcript_text["dialogues"]
        duration = transcript_text["duration"]


        ai_data = analyze_call_with_ai(dialogues)
        if ai_data:
            sentiment = ai_data.get("Sentiment", "Neutral").capitalize()
            if sentiment in sentiment_counts:
                sentiment_counts[sentiment] += 1

            conversation_analysis["Conversation Flow Score"] += safe_float(ai_data.get("Conversation Flow Score", 0))
            conversation_analysis["Question-to-Statement Ratio"] += safe_float(ai_data.get("Question-to-Statement Ratio", 0))
            conversation_analysis["Resolution Path Efficiency"] += safe_float(ai_data.get("Resolution Path Efficiency", 0))

        # ✅ Extract conversation & sentiment metrics from JSON
        extracted_metrics = extract_call_metrics(transcript_text)
        # 🔹 Send transcript to AI for sentiment classification
        ai_prompt = f"""
        Analyze the following customer service call and determine its overall sentiment:
        - Options: "Positive", "Neutral", or "Negative"
        - Consider customer tone, agent response, and resolution.

        - Rate flow smoothness (1-100)
    - Count question vs. statement ratio
    - Detect key topics
    - Identify resolution path efficiency

        Transcript:
        {transcript_text}

        Provide only one word as output: "Positive", "Neutral", or "Negative".

        Also provide conversational: 
        Provide output in JSON format:
        {{
            "Sentiment": "Positive | Neutral | Negative",
            "Conversation Flow Score": (1-100),
            "Question-to-Statement Ratio": (float),
            "Resolution Path Efficiency": (1-100)
        }}
        """
        response = llm.invoke(ai_prompt)

        # ✅ Extract AI-assigned sentiment
        try:
            ai_data = json.loads(response.content.strip())  # ✅ Parse JSON string
            sentiment = ai_data.get("sentiment", "Neutral").capitalize()
        except json.JSONDecodeError:
            print(f"⚠ Error parsing AI response for {file_key}: {response.content.strip()}")
            sentiment = "Neutral"
            ai_data = {"conversation flow score": 50, "question-to-statement ratio": 0.5, "resolution path efficiency": 50}

        if sentiment in ["Positive", "Neutral", "Negative"]:
            sentiment_counts[sentiment] += 1
        else:
            print(f"⚠ Unexpected sentiment received: {sentiment} for {file_key}")
    

    positive_calls = sentiment_counts["Positive"]
    csat_score = round((positive_calls / total_calls) * 100, 2) if total_calls else 0

    # ✅ NPS (Net Promoter Score)
    promoters = sentiment_counts["Positive"]  # Assume promoters are positive sentiment calls
    detractors = sentiment_counts["Negative"]  # Assume detractors are negative sentiment calls
    nps_score = round(((promoters - detractors) / total_calls) * 100, 2) if total_calls else 0

    agent_performance = {
        "Total Calls Handled": total_calls,
        "Average Call Duration": f"{average_duration} seconds",
        "CSAT (Customer Satisfaction Score)": f"{csat_score}%",
    }

    conversation_analysis["Conversation Flow Score"] += safe_float(ai_data.get("Conversation Flow Score", 0))
    conversation_analysis["Question-to-Statement Ratio"] += safe_float(ai_data.get("Question-to-Statement Ratio", 0))
    conversation_analysis["Resolution Path Efficiency"] += safe_float(ai_data.get("Resolution Path Efficiency", 0))

        # ✅ Store JSON-extracted conversation analysis
    # conversation_analysis["Speech-to-Silence Ratio"] += extracted_metrics["Speech-to-Silence Ratio"]
    # conversation_analysis["Interruption Rate"] += extracted_metrics["Interruption Rate"]
    # conversation_analysis["Hold Time Analysis"] += extracted_metrics["Hold Time Analysis"]

    # customer_sentiment["Escalation Risk Score"] += extracted_metrics.get("Escalation Risk Score", 0)

    # customer_sentiment["Frustration Score"] += extracted_metrics.get("Frustration score", 0)


    # ✅ Compute Majority Sentiment
    total_sentiments = sum(sentiment_counts.values())
    if total_sentiments > 0:
        for sentiment in sentiment_counts:
            customer_sentiment["Positive"] = round((sentiment_counts["Positive"] / total_sentiments) * 100, 2)
            customer_sentiment["Neutral"] = round((sentiment_counts["Neutral"] / total_sentiments) * 100, 2)
            customer_sentiment["Negative"] = round((sentiment_counts["Negative"] / total_sentiments) * 100, 2)

            customer_sentiment["Overall Sentiment Score"] = round(
    (customer_sentiment["Positive"] + customer_sentiment["Neutral"] + customer_sentiment["Negative"]) / 3, 2
)
    # ✅ Generate Weekly Summary Using GPT
    weekly_summary_prompt = f"""
    Generate a professional weekly call analysis based on the following data:

    - Total Calls: {total_calls}
    - Sentiment Breakdown:
      - Overall Sentiment Score: {customer_sentiment["Overall Sentiment Score"]}%

      - Positive: {customer_sentiment["Positive"]}%
- Neutral: {customer_sentiment["Neutral"]}%
- Negative: {customer_sentiment["Negative"]}%


    Summarize trends and overall customer sentiment.
    """
    weekly_summary_response = llm.invoke(weekly_summary_prompt)

    print(f"📝 AI Sentiment Response for {file_key}: {response.content.strip()}") 
    weekly_summary_paragraph = weekly_summary_response.content.strip()

    return weekly_summary_paragraph, agent_performance, customer_sentiment, conversation_analysis, date_range

    # ✅ Agent Performance Metrics
    

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

    Transcript:  
    {transcript}

    Metrics & Analysis:  
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
            dialogues, duration = extract_dialogues_and_duration(transcript_data)

            # Debugging logs
            print(f"✅ Processed transcript: {file_key}")
            print(f"🗣 Dialogues (First 500 chars): {dialogues[:500]}")
            print(f"⏳ Call Duration: {duration} seconds")

            transcript_texts[file_key] = {"dialogues": dialogues, "duration": duration}
        except json.JSONDecodeError:
            print(f"❌ Error parsing JSON file: {file_key}")
            continue

    return transcript_texts

# ✅ Analyze and store results
def analyze_and_store():
    transcripts = fetch_transcripts()
    if not transcripts:
        print("No transcripts found.")
        return

    date_range = start_and_end_date()
    analysis_filename = f"weekly_report_{date_range['from_date']}to{date_range['to_date']}.pdf"
    s3_analysis_path = f"keywords-bucket/{analysis_filename}"

    # ✅ Generate Weekly Summary & Metrics
    weekly_summary_paragraph, agent_performance, customer_sentiment, conversation_analysis, date_range = generate_weekly_content(transcripts, date_range)

    # ✅ Save Weekly Summary PDF
    save_weekly_pdf(weekly_summary_paragraph, agent_performance, customer_sentiment, conversation_analysis, "analysis_results.pdf")
    s3_client.upload_file("analysis_results.pdf", "keywords-bucket", analysis_filename)
    print(f"✅ Successfully uploaded weekly report to S3: s3://{s3_analysis_path}")

if __name__ == "__main__":
    analyze_and_store()