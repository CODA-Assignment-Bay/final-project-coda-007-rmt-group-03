import great_expectations as ge
from pyspark.sql import SparkSession

spark = SparkSession.builder.appName("GE Validate Clean Data").getOrCreate()

subjects = [
    "pendidikan_agama", "pendidikan_pancasila", "bahasa_indonesia",
    "matematika", "bahasa_inggris", "fisika", "kimia", "biologi",
    "sosiologi", "ekonomi", "geografi", "pjok", "informatika", "seni_dan_prakarya"
]

df = spark.read.parquet("/opt/airflow/data/processed/student_academic_records_clean.parquet")
ge_df = ge.dataset.SparkDFDataset(df)

# Validations
for s in subjects:
    ge_df.expect_column_values_to_be_between(s, min_value=0, max_value=100)
    ge_df.expect_column_values_to_not_be_null(s)

ge_df.expect_table_row_count_to_be_between(min_value=1, max_value=None)

result = ge_df.validate()
print(result)

if not result["success"]:
    raise ValueError("❌ Data validation failed in clean stage")
