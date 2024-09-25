import os
import pandas as pd
import joblib
from sklearn.metrics import mean_squared_error
import mlflow
import mlflow.sklearn


def score_model(model_folder, data_folder):
    with mlflow.start_run(run_name="model_scoring", nested=True) as run:
        # Load the model
        model_file_path = os.path.join(model_folder, "housing_model.pkl")
        model = joblib.load(model_file_path)

        # Log the loaded model
        mlflow.sklearn.log_model(model, "loaded_model")

        # Load the test dataset
        test_data_path = os.path.join(data_folder, "housing_data.csv")
        test_data = pd.read_csv(test_data_path)

        # Define the feature columns as they were during training
        feature_columns = [
            "longitude",
            "latitude",
            "housing_median_age",
            "total_rooms",
            "total_bedrooms",
            "population",
            "households",
            "median_income",
            "ocean_proximity",
        ]

        # Ensure all feature columns exist in the test data
        test_features = test_data[feature_columns]

        # Make predictions using the model
        predictions = model.predict(test_features)

        # Calculate metrics
        rmse = mean_squared_error(
            test_data["median_house_value"], predictions, squared=False
        )

        # Log predictions and metrics
        mlflow.log_metric("rmse", rmse)

        # Add predictions to the test data
        test_data["predicted_median_house_value"] = predictions

        # Save the predictions to a new CSV file
        output_file_path = os.path.join(data_folder, "predictions.csv")
        test_data.to_csv(output_file_path, index=False)

        print(f"Predictions saved to {output_file_path}")
