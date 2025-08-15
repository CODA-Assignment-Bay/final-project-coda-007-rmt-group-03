import os
import logging
import pandas as pd
import streamlit as st
from dotenv import load_dotenv
from openai import OpenAI

from query_runner import run_sql_query
from chart_suggester import suggest_chart
from agent import prompt_to_sql
from chart_builder import render_chart
from ai_context import CONTEXT

# ======== Konfigurasi dasar ========
logging.basicConfig(level=logging.INFO)
st.set_page_config(page_title="✨ Tanya AI", layout="wide")
st.title("✨ Tanya AI")

# ======== Load API Key ========
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    st.error("❌ OPENAI_API_KEY tidak ditemukan di .env")
    st.stop()

client = OpenAI(api_key=OPENAI_API_KEY)

# ======== Caching ========
@st.cache_data(show_spinner=False)
def cache_prompt_to_sql(prompt: str) -> str:
    return prompt_to_sql(prompt)

@st.cache_data(show_spinner=False)
def cache_run_sql(sql: str) -> pd.DataFrame:
    return run_sql_query(sql)

@st.cache_data(show_spinner=False)
def cache_generate_recommendation(df_summary: str, user_prompt: str) -> str:
    """Minta AI memberikan rekomendasi berdasarkan data & konteks proyek."""
    prompt = f"""
    {CONTEXT}

    Pertanyaan: {user_prompt}
    Ringkasan data: {df_summary}

    Buat rekomendasi singkat, praktis, dan relevan dengan tujuan di atas.
    Jawab dalam bahasa Indonesia.
    """
    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.3
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        return f"❌ Gagal membuat rekomendasi: {e}"

# ======== Input pengguna ========
user_input = st.text_input("Tulis pertanyaanmu:", "")

if user_input.strip():
    # Jalankan proses jika user_input berubah
    if "last_input" not in st.session_state or st.session_state.last_input != user_input:
        st.session_state.last_input = user_input

        # 1. Konversi prompt ke SQL
        with st.spinner("Menganalisa pertanyaan..."):
            try:
                sql = cache_prompt_to_sql(user_input)
                if not sql.strip():
                    st.error("❌ Gagal menghasilkan query SQL.")
                    st.stop()
                st.session_state.sql = sql
            except Exception as e:
                st.error(f"❌ Kesalahan saat membuat SQL: {e}")
                st.stop()

        # 2. Jalankan query SQL
        with st.spinner("Mengambil data dari database..."):
            try:
                df = cache_run_sql(sql)
                st.session_state.df = df
                st.session_state.chart_type = suggest_chart(df, user_input)
                st.success("✅ Data berhasil diambil!")
            except Exception as e:
                st.error(f"❌ Gagal menjalankan query: {e}")
                st.stop()

    # Ambil dari session_state
    sql = st.session_state.sql
    df = st.session_state.df
    default_chart_type = st.session_state.chart_type

    # ======== Tampilkan Query & DataFrame ========
    with st.expander("📄 Query SQL & Data", expanded=False):
        st.code(sql, language="sql")
        st.dataframe(df, use_container_width=True)

    # ======== Chart Section ========
    st.subheader("📈 Visualisasi Data")
    st.info(f"📊 Rekomendasi chart: **{default_chart_type}**")
    
    chart_options = [
        "table", "bar", "line", "scatter", "area", "pie", "box",
        "violin", "histogram", "strip", "density_heatmap", "radar"
    ]
    selected_chart = st.selectbox(
        "Pilih jenis chart", 
        chart_options,
        index=chart_options.index(default_chart_type) if default_chart_type in chart_options else 0
    )

    render_chart(df, selected_chart)

    # ======== Rekomendasi AI ========
    with st.spinner("🔍 Menganalisis data untuk rekomendasi..."):
        df_summary = df.describe(include="all").to_string()
        recommendation = cache_generate_recommendation(df_summary, user_input)

    st.subheader("💡 Rekomendasi")
    st.write(recommendation)
