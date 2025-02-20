from langchain.prompts import PromptTemplate
from langchain.output_parsers import StructuredOutputParser, ResponseSchema

response_schemas = [
    ResponseSchema(name="call_type", description="Type of call (e.g., Complaint, Support)"),
    ResponseSchema(name="sentiment", description="Overall sentiment (Positive, Neutral, Negative)"),
    ResponseSchema(name="emotion_detection", description="Detected emotions"),
]

output_parser = StructuredOutputParser.from_response_schemas(response_schemas)
format_instructions = output_parser.get_format_instructions()

call_analysis_prompt = PromptTemplate(
    template="""
    You are an AI call analyst. Analyze the following call transcript and extract:
    
    - Call Type
    - Sentiment Analysis
    - Emotion Detection
    
    Transcript:
    {transcript}
    
    {format_instructions}
    """,
    input_variables=["transcript"],
    partial_variables={"format_instructions": format_instructions},
)
