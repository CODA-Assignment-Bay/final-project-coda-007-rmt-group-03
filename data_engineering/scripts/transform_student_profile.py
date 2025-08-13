# transform_student_profile.py v2
from pyspark.sql import SparkSession
from pyspark.sql.functions import col, avg, expr, greatest, round as spark_round, from_json
from pyspark.sql.types import ArrayType, StringType

# Mapping sub_cluster -> cluster
cluster_subject_groups = {
    "Saintek": {
        "saintek_mipa": ["matematika", "fisika", "kimia", "biologi", "informatika"],
        "saintek_kesehatan": ["biologi", "kimia", "pjok"],
        "saintek_teknik": ["matematika", "fisika", "informatika"],
        "saintek_pertanian_peternakan": ["biologi", "kimia", "geografi"]
    },
    "Soshum": {
        "soshum_ekonomi_bisnis": ["ekonomi", "matematika", "bahasa_inggris"],
        "soshum_sosial_politik": ["sosiologi", "geografi", "pendidikan_pancasila"],
        "soshum_hukum": ["pendidikan_pancasila", "bahasa_indonesia", "sosiologi"],
        "soshum_psikologi": ["biologi", "sosiologi", "bahasa_inggris"]
    },
    "Bahasa_Sastra": {
        "bahasa_sastra": ["bahasa_indonesia", "bahasa_inggris"]
    },
    "Seni_Kreatif": {
        "seni_desain_kreatif": ["seni_dan_prakarya"]
    },
    "Olahraga_Kesehatan": {
        "olahraga_kesehatan": ["pjok", "biologi"]
    },
    "Keagamaan_Moral": {
        "keagamaan_moral": ["pendidikan_agama", "pendidikan_pancasila"]
    }
}

spark = SparkSession.builder.appName("Student Datamart Builder").getOrCreate()

# 1. Load student_clusters parquet
df_clusters = spark.read.parquet("/opt/airflow/data/processed/student_clusters.parquet")

# 2️. Group by student_id dan hitung rata-rata semua *_avg
avg_cols = [c for c in df_clusters.columns if c.endswith("_avg")]

df_avg = df_clusters.groupBy("student_id").agg(
    *[spark_round(avg(col(c)), 2).alias(c) for c in avg_cols]
)

# 3️. Tentukan sub_cluster berdasarkan nilai tertinggi
expr_case = "CASE " + " ".join(
    [f"WHEN {c} = greatest({', '.join(avg_cols)}) THEN '{c.replace('_avg', '')}'"
     for c in avg_cols]
) + " END"
df_avg = df_avg.withColumn("sub_cluster", expr(expr_case))

# 4. Tentukan cluster dari sub_cluster
sub_to_cluster = {}
for cluster_name, subs in cluster_subject_groups.items():
    for sub in subs.keys():
        sub_to_cluster[sub] = cluster_name

cluster_case_expr = "CASE " + " ".join(
    [f"WHEN sub_cluster = '{sub}' THEN '{cluster}'"
     for sub, cluster in sub_to_cluster.items()]
) + " END"
df_avg = df_avg.withColumn("cluster", expr(cluster_case_expr))

# 5. Load students.csv (pakai delimiter '|')
df_students = spark.read.option("sep", "|").csv("/opt/airflow/data/students.csv", header=True, inferSchema=True)

# 5b. Load student_academic_records.csv (pakai delimiter ',') dan ambil kolom tertentu saja
df_academic = spark.read.option("sep", ",").csv(
    "/opt/airflow/data/student_academic_records.csv", header=True, inferSchema=True
).select("student_id", "school_id")


# 6. Join
df_final = (
    df_avg
    .join(df_students, on="student_id", how="left")
    .join(df_academic, on="student_id", how="left")  # join school_id juga
)

# 7. Simpan output
df_final.write.mode("overwrite").parquet("/opt/airflow/data/processed/student_profile.parquet")

print("✅ Datamart student_profile berhasil dibuat.")
spark.stop()
