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
import re
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



def extract_dialogues_and_duration(transcript_data):
    job_name = transcript_data.get("jobName", "Unknown Job")  
    first_id = re.match(r"^\d+", job_name)  # Extract only first numeric part
    agent_number = first_id.group(0) if first_id else "Unknown"

    results = transcript_data.get("results", {})
    audio_segments = results.get("audio_segments", [])

    if not audio_segments:
        return agent_number, "No transcript available.", 0.0

    dialogues = []
    total_duration = 0.0

    for segment in audio_segments:
        speaker = segment.get("speaker_label", "Unknown Speaker")
        text = segment.get("transcript", "").strip()

        try:
            start_time = float(segment.get("start_time", "0.0"))
            end_time = float(segment.get("end_time", "0.0"))
            duration = round(end_time - start_time, 2)
            total_duration += duration
        except ValueError:
            print(f"⚠ Error converting time values in segment: {segment}")
            continue

        if text:
            dialogues.append(f"{speaker}: {text} (Duration: {duration}s)")

    return agent_number, "\n".join(dialogues), round(total_duration, 2)





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


def generate_weekly_content(transcripts_by_agent, date_range):
    grouped_analysis = {}

    for agent_number, transcripts in transcripts_by_agent.items():
        total_calls = len(transcripts)
        total_duration = sum(t["duration"] for t in transcripts)
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
        for file_key, transcript_text in enumerate(transcripts):  # transcripts is a list now
            dialogues = transcript_text["dialogues"]
            duration = transcript_text["duration"]

            ai_data = analyze_call_with_ai(dialogues)  # Keeping AI call as in your function
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
            {dialogues}

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
                sentiment = ai_data.get("Sentiment", "Neutral").capitalize()
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

        # ✅ Store analysis under the agent number
        grouped_analysis[agent_number] = {
            "Agent Performance": agent_performance,
            "Sentiment Breakdown": sentiment_counts,
            "Conversation Analysis": conversation_analysis
        }

    return grouped_analysis

    

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
    transcripts_by_agent = {}

    if "Contents" not in response:
        return transcripts_by_agent

    for obj in response["Contents"]:
        file_key = obj["Key"]
        if not file_key.endswith(".json"):
            continue

        file_obj = s3_client.get_object(Bucket=BUCKET_NAME, Key=file_key)
        file_content = file_obj["Body"].read().decode("utf-8")

        try:
            transcript_data = json.loads(file_content)
            agent_number, dialogues, duration = extract_dialogues_and_duration(transcript_data)

            print(f"✅ Processed transcript for Agent {agent_number}")

            if agent_number not in transcripts_by_agent:
                transcripts_by_agent[agent_number] = []

            transcripts_by_agent[agent_number].append({"dialogues": dialogues, "duration": duration})

        except json.JSONDecodeError:
            print(f"❌ Error parsing JSON file: {file_key}")
            continue

    return transcripts_by_agent

def save_grouped_pdf(grouped_analysis, transcripts_by_agent, filename):
    doc = SimpleDocTemplate(filename, pagesize=letter, leftMargin=72, rightMargin=72, topMargin=72, bottomMargin=72)
    styles = getSampleStyleSheet()
    elements = []

    for agent_number, data in grouped_analysis.items():
        elements.append(Paragraph(f"📌 Agent Number: {agent_number}", styles["Heading2"]))
        elements.append(Spacer(1, 12))

        elements.append(Paragraph("🔹 Agent Performance", styles["Heading3"]))
        agent_table_data = [["Metric", "Value"]] + [[k, str(v)] for k, v in data.get("Agent Performance", {}).items()]
        table = Table(agent_table_data, colWidths=[250, 150])
        table.setStyle(TableStyle([
            ('GRID', (0, 0), (-1, -1), 1, colors.black),
            ('BACKGROUND', (0, 0), (-1, 0), colors.grey),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.white),
        ]))
        elements.append(table)
        elements.append(Spacer(1, 20))

        # ✅ Sentiment Analysis
        if "Sentiment Breakdown" in data:
            elements.append(Paragraph("🔹 Sentiment Analysis", styles["Heading3"]))
            sentiment_table_data = [["Sentiment", "Count"]] + [[k, str(v)] for k, v in data["Sentiment Breakdown"].items()]
            table = Table(sentiment_table_data, colWidths=[250, 150])
            table.setStyle(TableStyle([
                ('GRID', (0, 0), (-1, -1), 1, colors.black),
                ('BACKGROUND', (0, 0), (-1, 0), colors.grey),
                ('TEXTCOLOR', (0, 0), (-1, 0), colors.white),
            ]))
            elements.append(table)
            elements.append(Spacer(1, 20))
        else:
            print(f"⚠ Warning: 'Sentiment Breakdown' missing for Agent {agent_number}")

        # ✅ Conversation Analysis
        if "Conversation Analysis" in data:
            elements.append(Paragraph("🔹 Conversation Analysis", styles["Heading3"]))
            conversation_table_data = [["Metric", "Value"]] + [[k, str(v)] for k, v in data["Conversation Analysis"].items()]
            table = Table(conversation_table_data, colWidths=[250, 150])
            table.setStyle(TableStyle([
                ('GRID', (0, 0), (-1, -1), 1, colors.black),
                ('BACKGROUND', (0, 0), (-1, 0), colors.grey),
                ('TEXTCOLOR', (0, 0), (-1, 0), colors.white),
            ]))
            elements.append(table)
            elements.append(Spacer(1, 20))
        else:
            print(f"⚠ Warning: 'Conversation Analysis' missing for Agent {agent_number}")

        # ✅ Per-Call Analysis
        if agent_number in transcripts_by_agent:
            elements.append(Paragraph("📞 Per-Call Analysis", styles["Heading3"]))
            for idx, transcript in enumerate(transcripts_by_agent[agent_number]):
                dialogues = transcript["dialogues"]
                duration = transcript["duration"]

                # Call-Level AI Analysis
                ai_data = analyze_call_with_ai(dialogues)

                elements.append(Paragraph(f"🗣 Call {idx + 1} (Duration: {duration} seconds)", styles["Heading4"]))
                elements.append(Paragraph(dialogues[:1000].replace("\n", "<br/>"), styles["BodyText"]))  # Truncated for readability
                elements.append(Spacer(1, 10))

                # Call Metrics Table
                if ai_data:
                    call_table_data = [["Metric", "Value"]] + [[k, str(v)] for k, v in ai_data.items()]
                    table = Table(call_table_data, colWidths=[250, 150])
                    table.setStyle(TableStyle([
                        ('GRID', (0, 0), (-1, -1), 1, colors.black),
                        ('BACKGROUND', (0, 0), (-1, 0), colors.grey),
                        ('TEXTCOLOR', (0, 0), (-1, 0), colors.white),
                    ]))
                    elements.append(table)
                    elements.append(Spacer(1, 20))

                elements.append(PageBreak())  # Page break after each call

    doc.build(elements)


# ✅ Analyze and store results
def analyze_and_store():
    transcripts_by_agent = fetch_transcripts()
    if not transcripts_by_agent:
        print("No transcripts found.")
        return

    date_range = start_and_end_date()
    grouped_analysis = generate_weekly_content(transcripts_by_agent, date_range)

    filename = f"weekly_report_{date_range['from_date']}to{date_range['to_date']}.pdf"
    save_grouped_pdf(grouped_analysis, transcripts_by_agent, filename)
    s3_client.upload_file(filename, "keywords-bucket", filename)
    print(f"✅ Successfully uploaded weekly report: s3://keywords-bucket/{filename}")


if __name__ == "__main__":
    analyze_and_store()