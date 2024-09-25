import mlflow
import pandas as pd
import os


def ingest_data(data_folder):
    with mlflow.start_run(run_name="data_ingestion", nested=True) as run:
        # Log the data folder being used
        mlflow.log_param("data_folder", data_folder)

        # Load the data
        data_path = os.path.join(data_folder, "housing_data.csv")
        data = pd.read_csv(data_path)

        # Log data details
        mlflow.log_metric("num_rows", data.shape[0])
        mlflow.log_metric("num_columns", data.shape[1])

        print(f"Ingested data from {data_path}")
        return data
