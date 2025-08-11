import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd


def detect_columns(df: pd.DataFrame):
    """
    Mengelompokkan kolom menjadi: time_cols, num_cols, cat_cols.
    Tidak melakukan konversi tipe data.
    """
    time_keywords = ["tahun", "year", "bulan", "tanggal", "waktu", "date", "time"]

    time_cols = [col for col in df.columns if any(kw in col.lower() for kw in time_keywords)]
    num_cols = [col for col in df.select_dtypes(include=["number"]).columns if col not in time_cols]
    cat_cols = df.select_dtypes(include=["object", "category"]).columns.tolist()

    return time_cols, num_cols, cat_cols


def render_chart(df: pd.DataFrame, chart_type: str):
    try:
        time_cols, num_cols, cat_cols = detect_columns(df)

        if chart_type == "table":
            st.dataframe(df, use_container_width=True)
            return

        fig = None  # deklarasi awal

        # LINE, BAR, AREA
        if chart_type in ["line", "bar", "area"]:
            x_candidates = time_cols + cat_cols
            x_col = x_candidates[0] if x_candidates else st.selectbox("Sumbu X", df.columns, key=f"x_{chart_type}")
            y_col = st.selectbox("Sumbu Y (nilai numerik)", num_cols, key=f"y_{chart_type}")
            color_candidates = [col for col in cat_cols if col != x_col and df[col].nunique() > 1]
            color_col = st.selectbox("Breakdown per kategori (opsional)", [None] + color_candidates,
                                     index=1 if color_candidates else 0,
                                     key=f"color_{chart_type}")

            if chart_type == "line":
                fig = px.line(df, x=x_col, y=y_col, color=color_col, markers=True, title="Line Chart (Tren)")
            elif chart_type == "bar":
                fig = px.bar(df, x=x_col, y=y_col, color=color_col, barmode="group", title="Bar Chart (Breakdown)")
            elif chart_type == "area":
                fig = px.area(df, x=x_col, y=y_col, color=color_col, line_group=color_col, title="Area Chart (Breakdown)")

            # Tambah data point label
            fig.update_traces(
                text=df[y_col],
                textposition="top center" if chart_type != "bar" else "outside",
                texttemplate="%{text}"
            )

        # SCATTER
        elif chart_type == "scatter":
            x_col = st.selectbox("Sumbu X", num_cols, key="x_scatter")
            y_col = st.selectbox("Sumbu Y", num_cols, key="y_scatter")
            color = st.selectbox("Warna", [None] + df.columns.tolist(), key="color_scatter")
            fig = px.scatter(df, x=x_col, y=y_col, color=color, text=df[y_col], title="Scatter Plot")
            fig.update_traces(textposition="top center", texttemplate="%{text}")

        # PIE
        elif chart_type == "pie":
            names = st.selectbox("Label", cat_cols, key="pie_names")
            values = st.selectbox("Nilai", num_cols, key="pie_values")
            fig = px.pie(df, names=names, values=values, title="Pie Chart")

        # BOX, VIOLIN, STRIP
        elif chart_type in ["box", "violin", "strip"]:
            y_col = st.selectbox("Nilai (numerik)", num_cols, key="y_box")
            x_col = st.selectbox("Kategori (opsional)", [None] + df.columns.tolist(), key="x_box")
            if chart_type == "box":
                fig = px.box(df, x=x_col, y=y_col, title="Box Plot")
            elif chart_type == "violin":
                fig = px.violin(df, x=x_col, y=y_col, box=True, title="Violin Plot")
            else:
                fig = px.strip(df, x=x_col, y=y_col, title="Strip Plot")

        # HISTOGRAM
        elif chart_type == "histogram":
            x_col = st.selectbox("Kolom Histogram", num_cols, key="x_hist")
            fig = px.histogram(df, x=x_col, title="Histogram")
            fig.update_traces(
                text=df[x_col],
                textposition="outside",
                texttemplate="%{text}"
            )

        # DENSITY HEATMAP
        elif chart_type == "density_heatmap":
            x_col = st.selectbox("Kolom X", num_cols, key="x_heat")
            y_col = st.selectbox("Kolom Y", num_cols, key="y_heat")
            fig = px.density_heatmap(df, x=x_col, y=y_col, title="Density Heatmap")

        # RADAR CHART
        elif chart_type == "radar":
            category_col = st.selectbox("Kolom kategori (axis)", cat_cols, key="radar_cat")
            value_col = st.selectbox("Kolom nilai numerik", num_cols, key="radar_val")
            group_col = st.selectbox("Kolom grup (opsional)", [None] + cat_cols, key="radar_group")

            if group_col:
                fig = px.line_polar(df, r=value_col, theta=category_col, color=group_col, line_close=True)
            else:
                fig = px.line_polar(df, r=value_col, theta=category_col, line_close=True)

            fig.update_traces(fill='toself')
            fig.update_layout(title="Radar Chart")

        # Tampilkan chart jika berhasil dibuat
        if fig:
            st.plotly_chart(fig, use_container_width=True)

    except Exception as e:
        st.error(f"❌ Gagal membuat chart: {e}")
