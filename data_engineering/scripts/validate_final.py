import great_expectations as ge
from pyspark.sql import SparkSession

spark = SparkSession.builder.appName("GE Validate Final Data").getOrCreate()

df = spark.read.parquet("/opt/airflow/data/processed/student_profile.parquet")
ge_df = ge.dataset.SparkDFDataset(df)

required_columns = ["student_id", "cluster", "sub_cluster"]
for col_name in required_columns:
    ge_df.expect_column_to_exist(col_name)

ge_df.expect_column_values_to_be_unique("student_id")
ge_df.expect_column_values_to_be_in_set(
    "cluster",
    ["Saintek", "Soshum", "Bahasa_Sastra", "Seni_Kreatif", "Olahraga_Kesehatan", "Keagamaan_Moral"]
)

result = ge_df.validate()
print(result)

if not result["success"]:
    raise ValueError("❌ Data validation failed in final stage")
