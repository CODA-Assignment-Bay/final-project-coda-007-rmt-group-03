import pandas as pd
import streamlit as st  # Dibutuhkan untuk debugging di Streamlit

# Mapping kata kunci ke jenis chart
CHART_KEYWORDS = {
    "pie": ["persentase", "proporsi", "prosentase", "komposisi"],
    "bar": ["perbandingan", "banding", "jumlah", "frekuensi", "total"],
    "line": ["trend", "tren","history", "perkembangan", "perubahan"],
    "scatter": ["hubungan", "korelasi", "relasi", "pengaruh"],
    "histogram": ["sebaran", "distribusi", "frekuensi", "kemunculan"],
    "box": ["distribusi","outlier", "boxplot", "variabilitas", "jangkauan", "rentang"],
    "heatmap": ["matrix", "korelasi", "heatmap", "hubungan antar variabel"],
    "violin": ["distribusi", "densitas", "violin plot"],
    "strip": ["distribusi", "strip chart", "scatter plot"],
    "radar": ["radar chart", "radar plot"],
}

def keyword_match(user_input: str) -> str:
    """Mencocokkan kata kunci dalam user_input untuk menentukan jenis chart."""
    user_input = user_input.lower()
    for chart_type, keywords in CHART_KEYWORDS.items():
        for keyword in keywords:
            if keyword in user_input:
                st.write(f"🔍 Keyword terdeteksi: '{keyword}' → Rekomendasi Chart: **{chart_type}**")
                return chart_type
    return ""

def suggest_chart(df: pd.DataFrame, user_input: str = "") -> str:
    """
    Menyarankan jenis chart berdasarkan keyword (prioritas utama),
    lalu fallback ke struktur DataFrame.
    """
    # Tahap 1: Deteksi dari keyword
    chart_from_keyword = keyword_match(user_input)
    if chart_from_keyword:
        return chart_from_keyword

    # Tahap 2: Fallback berdasarkan struktur DataFrame
    num_cols = df.select_dtypes(include=["number"]).columns
    cat_cols = df.select_dtypes(include=["object", "category"]).columns

    st.write("📊 Tidak ada keyword cocok, fallback ke struktur DataFrame:")
    st.write(f"📌 Kolom numerik: {list(num_cols)}")
    st.write(f"📌 Kolom kategorik: {list(cat_cols)}")

    if len(num_cols) >= 2:
        return "scatter"
    elif len(num_cols) == 1 and len(cat_cols) >= 1:
        return "bar"
    elif len(cat_cols) >= 1:
        return "pie"
    else:
        return "table"
