import os
import psycopg2
from psycopg2 import sql
from dotenv import load_dotenv

load_dotenv()

def connect_db():
    """Koneksi ke Postgres."""
    return psycopg2.connect(
        host=os.getenv("NEONDB_HOST"),
        port=os.getenv("NEONDB_PORT"),
        database=os.getenv("NEONDB_NAME"),
        user=os.getenv("NEONDB_USER"),
        password=os.getenv("NEONDB_PASSWORD")
    )

def get_schema_doc() -> str:
    """Menghasilkan dokumentasi schema untuk semua tabel & view di public."""
    with connect_db() as conn, conn.cursor() as cursor:
        cursor.execute("""
            SELECT table_name, table_type
            FROM information_schema.tables
            WHERE table_schema = 'public'
            ORDER BY table_name
        """)
        tables = cursor.fetchall()

        schema_doc = []

        for table_name, table_type in tables:
            schema_doc.append(f"Table: {table_name} ({table_type})")

            cursor.execute("""
                SELECT column_name, data_type, ordinal_position
                FROM information_schema.columns
                WHERE table_schema = 'public' AND table_name = %s
                ORDER BY ordinal_position
            """, (table_name,))
            columns = cursor.fetchall()

            for col_name, data_type, pos in columns:
                comment = None
                try:
                    cursor.execute(sql.SQL("""
                        SELECT pg_catalog.col_description(
                            (quote_ident(%s) || '.' || quote_ident(%s))::regclass::oid, %s
                        )
                    """), ('public', table_name, pos))
                    result = cursor.fetchone()
                    if result:
                        comment = result[0]
                except Exception:
                    comment = None

                schema_doc.append(f"- {col_name} ({data_type}): {comment or 'No description'}")

            schema_doc.append("")

        return "\n".join(schema_doc)

def save_schema_to_file(filename: str = "schema_doc.txt"):
    """Menyimpan hasil schema ke file .txt."""
    schema_text = get_schema_doc()
    with open(filename, "w", encoding="utf-8") as f:
        f.write(schema_text)
    print(f"✅ Schema berhasil disimpan ke {filename}")

if __name__ == "__main__":
    save_schema_to_file()
