import os
import re
import logging
from dotenv import load_dotenv
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from get_schema import get_schema_doc
from llm_factory import get_llm  # Modular inisialisasi LLM

# Load env
load_dotenv()

# Logging
logging.basicConfig(level=logging.INFO)

# Konfigurasi schema file cache
SCHEMA_CACHE_FILE = "schema_doc.txt"

def load_schema(refresh: bool = False) -> str:
    """
    Memuat schema database dari cache atau langsung dari NeonDB.
    """
    if not refresh and os.path.exists(SCHEMA_CACHE_FILE):
        logging.info("Memuat schema dari cache...")
        with open(SCHEMA_CACHE_FILE, "r", encoding="utf-8") as f:
            return f.read()
    else:
        logging.info("Mengambil schema dari NeonDB...")
        doc = get_schema_doc()
        with open(SCHEMA_CACHE_FILE, "w", encoding="utf-8") as f:
            f.write(doc)
        return doc

# Load schema dari cache / DB
TABLE_DOC = load_schema()

# Inisialisasi LLM dari modul eksternal
llm = get_llm(model_name="openai")  # Bisa diganti sesuai provider

# Prompt template untuk Postgres NeonDB
prompt_template = PromptTemplate.from_template(
    """Gunakan dokumentasi tabel berikut untuk membuat query SQL Postgres:
{schema_doc}

Ubah pertanyaan berikut menjadi query SQL Postgres yang kompatibel dengan NeonDB.

Prioritaskan untuk membaca data dari tabel `student_profile_datamart` jika relevan.
Jika ada pertanyaan terkait time series, gunakan tabel `student_clusters_history sebagai sumber data.
Jika dari kedua table di atas belum didapatkan data yang relevan, silakan join ke table lain.

Gunakan nama tabel persis seperti yang ada di dokumentasi (tanpa project ID atau dataset ID).

Hanya kembalikan query SQL tanpa format Markdown atau penjelasan apa pun.

Pertanyaan: {question}"""
)

# Buat LLMChain dengan schema sebagai partial input
sql_chain = LLMChain(
    llm=llm,
    prompt=prompt_template.partial(schema_doc=TABLE_DOC)
)

def clean_sql_output(text: str) -> str:
    """
    Membersihkan output LLM dari tag Markdown atau format tambahan.
    """
    text = re.sub(r"^```sql\s*", "", text.strip(), flags=re.IGNORECASE)
    text = re.sub(r"\s*```$", "", text.strip())
    return text.strip()

def prompt_to_sql(question: str) -> str:
    """
    Mengubah pertanyaan natural language menjadi query SQL Postgres.
    """
    try:
        raw_output = sql_chain.run(question)
        logging.info(f"Output SQL mentah: {raw_output}")
        return clean_sql_output(raw_output)
    except Exception as e:
        logging.error(f"Gagal mengonversi prompt ke SQL: {e}")
        return "-- Gagal menghasilkan query SQL --"

# Tes mandiri
if __name__ == "__main__":
    prompt = "Tampilkan 10 siswa dengan nilai tertinggi"
    print("Query SQL yang dihasilkan:")
    print(prompt_to_sql(prompt))
