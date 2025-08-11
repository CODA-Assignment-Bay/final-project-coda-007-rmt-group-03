# 🎓 AI-Powered Student Talent & Interest Identification

## 📌 Latar Belakang
Pendidikan adalah salah satu pilar utama dalam pembangunan berkelanjutan.  
Indonesia, dalam upaya mencapai *Sustainable Development Goals (SDG) 4: Quality Education*, masih menghadapi tantangan pemerataan akses pendidikan dan personalisasi pembelajaran.  
Salah satu pendekatan inovatif untuk meningkatkan kualitas pendidikan adalah **mengidentifikasi minat dan bakat siswa secara dini** agar dapat diarahkan ke jalur pendidikan atau karier yang sesuai.

Dengan memanfaatkan data akademik, aktivitas ekstrakurikuler, dan preferensi siswa, proyek ini membangun sistem berbasis **AI** untuk membantu guru, konselor, dan pembuat kebijakan mengambil keputusan berbasis data.

---

## ❓ Problem Statement (SMART Framework)
- **Specific**: Mengidentifikasi minat dan bakat siswa berdasarkan nilai akademik, hobi, dan kegiatan ekstrakurikuler.
- **Measurable**: Sistem mampu mengelompokkan siswa ke dalam minimal **3 cluster** profil minat-bakat dengan akurasi interpretasi > 80% (berdasarkan validasi internal).
- **Achievable**: Menggunakan data dummy representatif 100 siswa yang disimpan dalam PostgreSQL JSONB dan diproses dengan pipeline otomatis.
- **Relevant**: Mendukung tujuan SDG 4 untuk menyediakan pendidikan berkualitas dan inklusif.
- **Time-bound**: Proyek diselesaikan dalam **5 hari** mulai 9 Agustus 2025.

---

## 🔍 Key Questions
1. Bagaimana pola distribusi nilai siswa pada berbagai mata pelajaran?
2. Apakah terdapat hubungan antara kegiatan ekstrakurikuler dan minat karier?
3. Bagaimana segmentasi siswa berdasarkan nilai akademik dan preferensi?
4. Rekomendasi karier atau jalur studi apa yang sesuai untuk tiap cluster siswa?

---

## 🎯 Tujuan Proyek
1. Membangun **ETL pipeline otomatis** dengan Apache Airflow untuk memproses data minat dan bakat siswa.
2. Melakukan **analisis eksplorasi data (EDA)** untuk memahami pola dan distribusi.
3. Menerapkan **clustering** untuk mengelompokkan siswa berdasarkan karakteristik akademik dan non-akademik.
4. Memvalidasi kualitas data menggunakan **Great Expectations**.
5. Membuat **datamart** untuk dikonsumsi oleh *Business Intelligence tools* (Looker Studio, Tableau) dan aplikasi **Streamlit** dengan integrasi **LangChain AI agent**.

---

## 📊 Exploratory Data Analysis (EDA)
EDA dilakukan untuk:
- Memeriksa distribusi nilai tiap mata pelajaran.
- Mengidentifikasi outlier dan missing value.
- Menganalisis keterkaitan antara hobi, ekskul, dan nilai akademik.
- Visualisasi awal dengan histogram, heatmap korelasi, dan boxplot per cluster.

---

## ⚙️ ETL Pipeline (Apache Airflow)
Pipeline dibagi menjadi tiga tahap utama:

### **1. Extract (`extract.py`)**
- Membaca dataset sumber (CSV/JSON).
- Menyimpan hasil ekstraksi dalam staging area.

### **2. Transform (`transform.py`)**
- **Data Cleaning**: 
  - Mengubah format tanggal lahir → usia.
  - Menangani missing value.
- **Transformation**:
  - Normalisasi nilai akademik.
  - Encoding fitur kategorikal.
- **Clustering**:
  - Menggunakan **K-Means** untuk segmentasi minat-bakat.
  - Menambahkan label cluster ke dataset.

### **3. Load (`load.py`)**
- Memvalidasi dataset menggunakan **Great Expectations**:
  - Struktur JSONB sesuai schema.
  - Nilai akademik berada dalam rentang 0–100.
  - Field wajib (`nama_siswa`, `tanggal_lahir`, `gender`) terisi.
- Memuat data ke **Neon PostgreSQL** (JSONB columns untuk data dinamis).

---

## 🗄️ Data Architecture

Source Data (CSV/JSON)
        ↓
[Extract] → Staging Area
        ↓
[Transform & Cluster]
        ↓
[Validation: Great Expectations]
        ↓
[Load to Neon PostgreSQL (JSONB)]
        ↓
Datamart (Aggregated Views)
        ↓
BI Tools / AI Dashboard

## 📦 Datamart
Datamart dibuat di PostgreSQL dengan tujuan mempermudah konsumsi data oleh:

Looker Studio → Dashboard interaktif minat-bakat siswa.

Tableau → Analisis visual lanjutan.

Streamlit + LangChain AI Agent → Chatbot rekomendasi minat karier.

Contoh tabel datamart:
CREATE VIEW datamart_siswa AS
SELECT
    id,
    nama_siswa,
    usia,
    gender,
    cluster,
    minat_karir,
    akademik -> 'Matematika' AS nilai_mtk,
    akademik -> 'IPA' AS nilai_ipa
FROM siswa;


## 🤖 AI Dashboard (Streamlit + LangChain)
Fitur utama:

Upload Data Siswa → Menampilkan profil dan cluster.

Chatbot AI → Memberikan rekomendasi jalur studi/karier berdasarkan profil siswa.

Visualisasi Dinamis → Radar chart, bar chart nilai per siswa, distribusi cluster.



# 📊 Student Dashboard with AI Assistant

## 📌 Deskripsi
Dashboard ini menampilkan analisis minat dan bakat siswa yang diambil dari **Neon PostgreSQL**, menampilkan visualisasi interaktif, dan menyediakan **AI Assistant** berbasis LangChain untuk memberikan rekomendasi jalur studi/karier.

---

## ⚙️ Fitur
- Menampilkan data siswa dan cluster minat-bakat
- Visualisasi interaktif dengan Plotly
- Chatbot AI untuk rekomendasi karier
- Query datamart dari Neon PostgreSQL

---

## 📂 Struktur Folder
.
├── notebooks/
│ └── eda.ipynb
├── queries/
│ └── datamart.sql
├── streamlit_app/
│ ├── app.py
│ └── utils.py
├── .env # DB & API Key
├── requirements.txt
└── README.md


pip install -r requirements.txt