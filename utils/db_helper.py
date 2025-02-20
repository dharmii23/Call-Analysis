import psycopg2
import json
from config import DB_CONFIG  # Make sure you have DB config

def save_analysis_to_db(call_id, analysis_result):
    """Saves call analysis result to the database."""
    try:
        conn = psycopg2.connect(**DB_CONFIG)
        cursor = conn.cursor()

        # Convert analysis result to JSON string
        analysis_json = json.dumps(analysis_result)

        # Insert data into call_analysis table
        cursor.execute("""
            INSERT INTO call_analysis (call_id, analysis_result, created_at)
            VALUES (%s, %s, NOW())
            ON CONFLICT (call_id) DO UPDATE 
            SET analysis_result = EXCLUDED.analysis_result, updated_at = NOW();
        """, (call_id, analysis_json))

        conn.commit()
        cursor.close()
        conn.close()

        print(f"✅ Analysis saved to DB for Call ID: {call_id}")

    except Exception as e:
        print(f"❌ Database error: {e}")
