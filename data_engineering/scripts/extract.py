# extract.py
from pyspark.sql import SparkSession
from typing import Dict, Any
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def create_spark_session(app_name: str = "Data Extraction") -> SparkSession:
    """
    Create or get a SparkSession.
    """
    spark = SparkSession.builder \
        .appName(app_name) \
        .config("spark.sql.session.timeZone", "UTC") \
        .getOrCreate()
    return spark

def extract_data(spark: SparkSession,
                 file_path: str,
                 file_format: str = "csv",
                 header: bool = True,
                 infer_schema: bool = True,
                 count_rows: bool = False,
                 **read_options) -> Any:
    """
    Read a single file into a Spark DataFrame.

    Args:
        spark: SparkSession instance.
        file_path: path to the file (CSV/Parquet/JSON).
        file_format: 'csv' (default), 'parquet', 'json', etc.
        header: use header if csv.
        infer_schema: infer schema if csv.
        count_rows: if True will call .count() and return (df, count).
        read_options: additional options forwarded to spark.read (e.g., sep, encoding).

    Returns:
        Spark DataFrame or tuple (df, count) if count_rows=True.
    """
    logger.info(f"Reading {file_path} (format={file_format})")
    reader = spark.read.format(file_format)

    # Set common options
    if file_format.lower() == "csv":
        reader = reader.option("header", str(header)).option("inferSchema", str(infer_schema))
    for k, v in read_options.items():
        reader = reader.option(k, v)

    df = reader.load(file_path)

    if count_rows:
        cnt = df.count()  # expensive action
        logger.info(f"Loaded {file_path} -> rows: {cnt}, cols: {len(df.columns)}")
        return df, cnt

    logger.info(f"Loaded {file_path} -> cols: {len(df.columns)} (row count skipped)")
    return df

def extract_multiple(spark: SparkSession,
                     paths: Dict[str, str],
                     file_format: str = "csv",
                     count_rows: bool = False,
                     **read_options) -> Dict[str, Any]:
    """
    Read multiple files and return a dict of DataFrames.
    Args:
        spark: SparkSession
        paths: dict of name -> path, e.g. {"students": "/opt/.../students.csv", "records": "..."}
        file_format: default 'csv'
        count_rows: if True, each value is (df, count)
        read_options: forwarded to spark.read
    Returns:
        dict name -> DataFrame (or (DataFrame, count) if count_rows=True)
    """
    results = {}
    for name, path in paths.items():
        try:
            results[name] = extract_data(spark, path, file_format=file_format, count_rows=count_rows, **read_options)
        except Exception as e:
            logger.exception(f"Failed reading {path}: {e}")
            raise
    return results

# Example usage when run as script (for local debug)
if __name__ == "__main__":
    spark = create_spark_session("extract_debug")
    paths = {
        "students": "/opt/airflow/data/students.csv",
        "records": "/opt/airflow/data/student_academic_records.csv"
    }
    dfs = extract_multiple(spark, paths, file_format="csv", count_rows=False)
    students_df = dfs["students"]
    records_df = dfs["records"]

    students_df.show(5, truncate=False)
    records_df.show(5, truncate=False)

    # stop spark when done (important in scripts/tasks)
    spark.stop()
