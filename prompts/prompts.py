from langchain.prompts import PromptTemplate
from langchain.output_parsers import StructuredOutputParser, ResponseSchema

response_schemas = [
    ResponseSchema(name="call_type", description="Categorization of the call (e.g., Billing Issue, Technical Support, General Inquiry)"),
    ResponseSchema(name="sentiment", description="Overall sentiment analysis of the call (Positive, Neutral, Negative)"),
    ResponseSchema(name="emotion_detection", description="Detected emotions such as frustration, anger, happiness, calmness"),
    ResponseSchema(name="conversation_flow", description="Measures smoothness and logical progression of dialogue"),
    ResponseSchema(name="speech_to_silence_ratio", description="Ratio of spoken words to silence during the call"),
    ResponseSchema(name="interruption_rate", description="How often either party interrupts the other"),
    ResponseSchema(name="question_to_statement_ratio", description="Ratio of questions asked by the agent vs. statements made"),
    ResponseSchema(name="customer_effort_score", description="Measures ease of issue resolution for the customer"),
    ResponseSchema(name="issue_detected", description="Extracts key issues mentioned by the customer, e.g., 'late delivery', 'broken product'")
]

output_parser = StructuredOutputParser.from_response_schemas(response_schemas)
format_instructions = output_parser.get_format_instructions()

call_analysis_prompt = PromptTemplate(
    template="""
    You are an AI-powered call analyst designed to assess customer interactions and extract meaningful insights.
    Analyze the following call transcript and provide insights based on the key metrics:
    
    **Key Components of Call Analysis:**
    
    1. **Call Categorization:** Identify whether the call is about Billing, Support, Product Issue, or General Inquiry.
    2. **Agent Performance Metrics:** Analyze the agent's performance including:
        - Total Calls Handled
        - Average Handling Time (AHT)
        - Call per Hour
        - Idle Time & Occupancy Rate
        - First Call Resolution (FCR)
        - Adherence to Script & Compliance
    3. **Customer Sentiment & Experience:**
        - Overall Sentiment (Positive, Neutral, Negative)
        - Emotion Detection (e.g., frustration, anger, happiness, calmness)
        - Customer Effort Score (CES) - Measures ease of issue resolution
        - Sentiment Shift Analysis - Detects changes in sentiment over the call
    4. **Conversation Analysis:**
        - Conversation Flow - Measures logical progression and coherence
        - Speech-to-Silence Ratio - Identifies if one party is dominating
        - Interruption Rate - Frequency of interruptions
        - Question-to-Statement Ratio - Ensures agents ask enough questions
        - Hold Time Analysis - Number and duration of holds
    5. **Issue & Intent Detection:**
        - Identifies key customer complaints (e.g., "defective product", "wrong billing", "late delivery")
        - Detects keywords and actionable insights
    
    **Transcript:**
    {transcript}
    
    {format_instructions}
    """,
    input_variables=["transcript"],
    partial_variables={"format_instructions": format_instructions},
)
