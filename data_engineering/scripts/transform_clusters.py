# transform_clusters.py
from pyspark.sql import SparkSession
from pyspark.sql.functions import col, greatest, round as spark_round, expr
from pyspark.sql.types import DoubleType
from functools import reduce

def transform_data(records_path, output_path, debug=False):
    spark = SparkSession.builder.appName("Student Thematic Mapping").getOrCreate()

    # Semua mata pelajaran yang digunakan
    subjects = [
        "pendidikan_agama",
        "pendidikan_pancasila",
        "bahasa_indonesia",
        "matematika",
        "bahasa_inggris",
        "fisika",
        "kimia",
        "biologi",
        "sosiologi",
        "ekonomi",
        "geografi",
        "pjok",
        "informatika",
        "seni_dan_prakarya"
    ]

    # Mapping cluster → sub_cluster → subjects
    cluster_subject_groups = {
        "Saintek": {
            "Saintek_MIPA": ["matematika", "fisika", "kimia", "biologi", "informatika"],
            "Saintek_Kesehatan": ["biologi", "kimia"],
            "Saintek_Teknik": ["matematika", "fisika", "informatika"],
            "Saintek_Pertanian_Peternakan": ["biologi", "kimia", "geografi"]
        },
        "Soshum": {
            "Soshum_Ekonomi_Bisnis": ["ekonomi", "matematika", "bahasa_inggris"],
            "Soshum_Sosial_Politik": ["sosiologi", "geografi", "pendidikan_pancasila"],
            "Soshum_Hukum": ["pendidikan_pancasila", "bahasa_indonesia"],
            "Soshum_Psikologi": ["biologi", "sosiologi", "bahasa_inggris"]
        },
        "Bahasa": {
            "Bahasa_Sastra": ["bahasa_indonesia", "bahasa_inggris"]
        },
        "Seni": {
            "Seni_Desain_Kreatif": ["seni_dan_prakarya"]
        },
        "Olahraga": {
            "Olahraga_Kesehatan": ["pjok","biologi"]
        },
        "Keagamaan": {
            "Keagamaan_Moral": ["pendidikan_agama", "pendidikan_pancasila"]
        }
    }

    # 1️⃣ Load data
    df = spark.read.parquet(records_path)
    df = df.dropDuplicates().na.drop(subset=["student_id"])

    # 2️⃣ Pastikan semua numeric
    for sub in subjects:
        df = df.withColumn(sub, col(sub).cast(DoubleType()))

    # 3️⃣ Drop baris yang semua nilainya null
    all_nulls_cond = reduce(lambda a, b: a & b, [col(s).isNull() for s in subjects])
    df = df.filter(~all_nulls_cond).na.fill(0, subset=subjects)

    # 4️⃣ Hitung rata-rata per sub_cluster
    subcluster_avg_cols = {}
    for cluster_name, subclusters in cluster_subject_groups.items():
        for subcluster_name, subs in subclusters.items():
            colname = subcluster_name.lower() + "_avg"
            subcluster_avg_cols[subcluster_name] = colname
            df = df.withColumn(
                colname,
                spark_round(sum([col(s) for s in subs]) / len(subs), 2)
            )

    # 5️⃣ Tentukan sub_cluster dengan nilai tertinggi
    avg_cols_list = list(subcluster_avg_cols.values())
    expr_case_sub = "CASE " + " ".join(
        [f"WHEN {colname} = greatest({', '.join(avg_cols_list)}) THEN '{subcluster_name}'"
         for subcluster_name, colname in subcluster_avg_cols.items()]
    ) + " END"
    df = df.withColumn("sub_cluster", expr(expr_case_sub))

    # 6️⃣ Tentukan cluster berdasarkan sub_cluster
    cluster_case_parts = []
    for cluster_name, subclusters in cluster_subject_groups.items():
        subcluster_list_str = ", ".join([f"'{sc}'" for sc in subclusters.keys()])
        cluster_case_parts.append(f"WHEN sub_cluster IN ({subcluster_list_str}) THEN '{cluster_name}'")
    cluster_mapping_expr = "CASE " + " ".join(cluster_case_parts) + " END"

    df = df.withColumn("cluster", expr(cluster_mapping_expr))

    # 7️⃣ Debug preview
    if debug:
        df.show(10, truncate=False)

    # 8️⃣ Simpan hasil
    df.select(
        "student_id", "record_date", "cluster", "sub_cluster", *avg_cols_list
    ).write.mode("overwrite").parquet(output_path)

    print(f"✅ Mapping cluster & sub_cluster selesai. Data disimpan di {output_path}")
    spark.stop()

if __name__ == "__main__":
    transform_data(
        records_path="/opt/airflow/data/processed/student_academic_records_clean.parquet",
        output_path="/opt/airflow/data/processed/student_clusters.parquet",
        debug=True
    )
