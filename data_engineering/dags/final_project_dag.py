import datetime as dt
from datetime import timedelta

from airflow import DAG
from airflow.operators.bash_operator import BashOperator
from airflow.operators.python_operator import PythonOperator


default_args = {
    'owner': 'group-03',
    'start_date': dt.datetime(2025, 1, 8),
    'retries': 1,
    'retry_delay': dt.timedelta(minutes=600),
}


with DAG('final-project-coda-007-rmt-group-03',
         default_args=default_args,
         schedule_interval='0 9 1 * *',
         catchup=False,
         ) as dag:

    python_extract = BashOperator(task_id='python_extract', bash_command='sudo -u airflow python /opt/airflow/scripts/extract.py')
    python_clean = BashOperator(task_id='python_clean', bash_command='sudo -u airflow python //opt/airflow/scripts/clean.py')
    python_transform_clusters = BashOperator(task_id='python_transform_clusters', bash_command='sudo -u airflow python /opt/airflow/scripts/transform_clusters.py')
    python_transform_student_profile = BashOperator(task_id='python_transform_student_profile', bash_command='sudo -u airflow python /opt/airflow/scripts/transform_student_profile.py')
    python_load = BashOperator(task_id='python_load', bash_command='sudo -u airflow python /opt/airflow/scripts/load.py')


python_extract >> python_clean >> python_transform_clusters >> python_transform_student_profile >> python_load