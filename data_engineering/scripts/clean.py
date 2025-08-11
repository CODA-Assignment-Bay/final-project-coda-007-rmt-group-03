# clean.py
from functools import reduce
from pyspark.sql import SparkSession
from pyspark.sql.functions import col, from_json, avg, regexp_replace, trim, when, to_json, struct
from pyspark.sql.types import DoubleType, StructType, StructField
import pyspark.sql.functions as F

# Inisialisasi Spark
spark = SparkSession.builder.appName("Validate Academic Records").getOrCreate()

# Subjects
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

academic_schema = StructType([StructField(s, DoubleType(), True) for s in subjects])

# Path file
records_path = "/opt/airflow/data/student_academic_records.csv"

# Load CSV dengan opsi agar parsing JSON aman
df = spark.read.option("header", True) \
               .option("multiLine", True) \
               .option("quote", "\"") \
               .option("escape", "\"") \
               .csv(records_path)

print("Schema awal:")
df.printSchema()

# Bersihkan JSON string di academic_scores
df = df.withColumn("academic_scores_clean", col("academic_scores"))
df = df.withColumn("academic_scores_clean", regexp_replace(col("academic_scores_clean"), r'""', r'"'))
df = df.withColumn("academic_scores_clean", regexp_replace(col("academic_scores_clean"), r"''", r'"'))
df = df.withColumn("academic_scores_clean", regexp_replace(col("academic_scores_clean"), r'^\s*"(.*)"\s*$', r'$1'))
df = df.withColumn("academic_scores_clean", regexp_replace(col("academic_scores_clean"), r"^\s*'(.*)'\s*$", r'$1'))
df = df.withColumn("academic_scores_clean", trim(col("academic_scores_clean")))

# Parse JSON
df = df.withColumn("scores_struct", from_json(col("academic_scores_clean"), academic_schema))

# Cek parsing gagal
parsing_failed_count = df.filter(col("scores_struct").isNull()).count()
print(f"Parsing gagal: {parsing_failed_count} rows")

# Ekstrak kolom subject
for s in subjects:
    df = df.withColumn(s, col("scores_struct")[s].cast(DoubleType()))

# Cek null per subject
null_counts = df.select([F.count(F.when(col(s).isNull(), s)).alias(s + "_nulls") for s in subjects])
null_counts.show(truncate=False)

# Imputasi nilai mean (opsional)
fill_map = {s: float(df.select(avg(col(s))).first()[0] or 0.0) for s in subjects}
df = df.na.fill(fill_map, subset=subjects)

# Buat ulang academic_scores bersih
df = df.withColumn("academic_scores_clean", to_json(struct(*[col(s) for s in subjects])))

# Simpan hasil bersih (opsional)
df.write.mode("overwrite").parquet("/opt/airflow/data/processed/student_academic_records_clean.parquet")
print("✅ Data bersih disimpan ke /opt/airflow/data/processed/student_academic_records_clean.parquet")
