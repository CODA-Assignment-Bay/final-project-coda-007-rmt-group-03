# load.py
import os
import json
import numpy as np
import pandas as pd
from sqlalchemy import create_engine
from dotenv import load_dotenv

def create_engine_from_env():
    load_dotenv()
    PG_USER = "neondb_owner"
    PG_PASSWORD = "npg_akq7JMdOG1RY"
    PG_HOST = "ep-black-flower-a1i22rs0-pooler.ap-southeast-1.aws.neon.tech"
    PG_PORT = 5432
    PG_DB = "neondb"
    PG_SSLMODE = "require"

    if not all([PG_USER, PG_PASSWORD, PG_HOST, PG_DB]):
        raise ValueError("❌ Database credentials belum lengkap di .env")

    return create_engine(
        f"postgresql+psycopg2://{PG_USER}:{PG_PASSWORD}@{PG_HOST}:{PG_PORT}/{PG_DB}?sslmode={PG_SSLMODE}"
    )

def convert_ndarray_to_str(df):
    """
    Ubah semua kolom yang berisi list atau numpy.ndarray menjadi string JSON
    Contoh: ["drawing", "coding", "traveling"]
    """
    for col in df.columns:
        if df[col].apply(lambda x: isinstance(x, (list, np.ndarray))).any():
            df[col] = df[col].apply(lambda x: json.dumps(list(x)) if isinstance(x, (list, np.ndarray)) else x)
    return df

def load_csv_to_postgres(engine, file_path, table_name, if_exists="replace", sep=","):
    print(f"📂 Membaca CSV: {file_path} (delimiter='{sep}')")
    df = pd.read_csv(file_path, sep=sep)
    df = convert_ndarray_to_str(df)
    print(f"🚀 Mengirim data ke tabel: {table_name}")
    df.to_sql(table_name, engine, if_exists=if_exists, index=False)
    print(f"✅ Tabel '{table_name}' selesai di-load")

def load_parquet_to_postgres(engine, file_path, table_name, if_exists="replace"):
    print(f"📂 Membaca Parquet: {file_path}")
    df = pd.read_parquet(file_path)
    df = convert_ndarray_to_str(df)
    print(f"🚀 Mengirim data ke tabel: {table_name}")
    df.to_sql(table_name, engine, if_exists=if_exists, index=False)
    print(f"✅ Tabel '{table_name}' selesai di-load")

if __name__ == "__main__":
    engine = create_engine_from_env()

    # Load student.csv
    load_csv_to_postgres(engine, "data/students.csv", "students", sep="|")

    # # Load student_academic_records.csv
    # load_csv_to_postgres(engine, "data/student_academic_records.csv", "student_academic_record_raw")
    
    # Load student_academic_records_clean.parquet
    load_parquet_to_postgres(engine, "data/processed/student_academic_records_clean.parquet", "student_academic_records")
    
    # Load student_clusters.parquet
    load_parquet_to_postgres(engine, "data/processed/student_clusters.parquet", "student_clusters_history")

    # Load student_profiles.parquet
    load_parquet_to_postgres(engine, "data/processed/student_profile.parquet", "student_profile_datamart")
