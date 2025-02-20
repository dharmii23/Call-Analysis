from airflow import DAG
from airflow.operators.python import PythonOperator
from datetime import datetime, timedelta
from tasks.fetch_audio import fetch_audio_from_kaleyra
from tasks.transcribe_audio import transcribe_audio_with_aws
from tasks.analyze_transcript import analyze_transcript  # Updated

# ✅ Corrected Default DAG Arguments
default_args = {
    'owner': 'airflow',
    'start_date': datetime.today() - timedelta(days=1),  # Always fetch yesterday’s recordings
    'retries': 1,
}

# ✅ Corrected DAG Definition
with DAG(
    'call_analysis_pipeline',
    default_args=default_args,
    schedule="0 0 * * *",  # ✅ Replaced `schedule_interval` with `schedule`
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

    analyze_transcript_task = PythonOperator(  # Merged Task
        task_id='analyze_transcript',
        python_callable=analyze_transcript
    )

    # ✅ Define Task Dependencies Correctly
    fetch_audio_task >> transcribe_audio_task >> analyze_transcript_task
