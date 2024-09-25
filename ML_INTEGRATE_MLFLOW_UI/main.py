import mlflow
import ingest_data
import train
import score


def main(data_folder, model_folder):
    # End any active MLFlow run
    mlflow.end_run()

    with mlflow.start_run(run_name="end_to_end_pipeline"):
        # Step 1: Data Ingestion
        mlflow.set_tag("stage", "data_ingestion")
        ingest_data.ingest_data(data_folder)

        # Step 2: Model Training
        mlflow.set_tag("stage", "model_training")
        train.train_model(data_folder)

        # Step 3: Model Scoring
        mlflow.set_tag("stage", "model_scoring")
        score.score_model(model_folder, data_folder)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="End-to-End MLFlow pipeline.")
    parser.add_argument("--data-folder", required=True, help="Path to the data folder.")
    parser.add_argument(
        "--model-folder", required=True, help="Path to the model folder."
    )
    args = parser.parse_args()

    main(args.data_folder, args.model_folder)
