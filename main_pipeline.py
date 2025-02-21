import os
from dotenv import load_dotenv
from airflow import DAG
from airflow.operators.python import PythonOperator
from datetime import datetime, timedelta
from tasks.fetch_audio import fetch_audio_from_kaleyra
from tasks.transcribe_audio import transcribe_audio_with_aws
from tasks.analyze_transcript import analyze_transcript

# Load environment variables
load_dotenv()

# Default DAG Arguments
default_args = {
    'owner': 'airflow',
    'start_date': datetime.today() - timedelta(days=1),
    'retries': 1,
}

with DAG(
    'call_analysis_pipeline',
    default_args=default_args,
    schedule="0 0 * * *",
    catchup=False
) as dag:
    
    fetch_audio_task = PythonOperator(
        task_id='fetch_audio',
        python_callable=fetch_audio_from_kaleyra
    )

    transcribe_audio_task = PythonOperator(
        task_id='transcribe_audio',
        python_callable=transcribe_audio_with_aws
    )

    analyze_transcript_task = PythonOperator(
        task_id='analyze_transcript',
        python_callable=analyze_transcript
    )

    fetch_audio_task >> transcribe_audio_task >> analyze_transcript_task
