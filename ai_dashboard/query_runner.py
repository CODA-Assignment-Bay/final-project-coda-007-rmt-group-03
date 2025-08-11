import os
import psycopg2
import pandas as pd
from psycopg2 import sql
from dotenv import load_dotenv
from typing import Optional

# Load environment variables
load_dotenv()

def connect_db():
    """Membuat koneksi ke NeonDB Postgres."""
    return psycopg2.connect(
        host=os.getenv("NEONDB_HOST"),
        port=os.getenv("NEONDB_PORT"),
        database=os.getenv("NEONDB_NAME"),
        user=os.getenv("NEONDB_USER"),
        password=os.getenv("NEONDB_PASSWORD")
    )

def clean_sql(sql_text: str) -> str:
    """Membersihkan query SQL dari prefix yang tidak perlu."""
    if not sql_text or not sql_text.strip():
        raise ValueError("Query SQL kosong. Pastikan pertanyaan dikonversi dengan benar.")
    
    sql_cleaned = sql_text.strip()
    if sql_cleaned.lower().startswith("sql"):
        sql_cleaned = sql_cleaned[3:].lstrip()
    
    return sql_cleaned

def run_sql_query(query: str) -> pd.DataFrame:
    """
    Menjalankan query SQL di NeonDB Postgres dan mengembalikan hasilnya sebagai DataFrame.

    Args:
        query (str): Query SQL yang akan dijalankan.

    Returns:
        pd.DataFrame: Hasil query dalam bentuk DataFrame.
    """
    sql_cleaned = clean_sql(query)

    try:
        with connect_db() as conn, conn.cursor() as cursor:
            cursor.execute(sql_cleaned)
            columns = [desc[0] for desc in cursor.description] if cursor.description else []
            results = cursor.fetchall() if cursor.description else []
            
            return pd.DataFrame(results, columns=columns)

    except psycopg2.Error as db_err:
        raise RuntimeError(f"Database error: {db_err}") from db_err
