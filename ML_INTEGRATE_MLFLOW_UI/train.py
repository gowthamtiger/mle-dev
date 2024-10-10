import os
import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.ensemble import RandomForestRegressor
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_squared_error
import mlflow
import mlflow.sklearn


def train_model(data_folder):
    with mlflow.start_run(run_name="model_training", nested=True) as run:

        # Load data
        data_path = os.path.join(data_folder, "housing_data.csv")
        data = pd.read_csv(data_path)

        # Define features and target
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
        target_column = "median_house_value"

        X = data[feature_columns]
        y = data[target_column]

        # Split the data
        X_train, X_valid, y_train, y_valid = train_test_split(
            X, y, test_size=0.2, random_state=42
        )

        # Log data split info
        mlflow.log_metric("train_size", len(X_train))
        mlflow.log_metric("valid_size", len(X_valid))

        # Define the preprocessing steps
        preprocessor = ColumnTransformer(
            transformers=[
                (
                    "num",
                    StandardScaler(),
                    [
                        "longitude",
                        "latitude",
                        "housing_median_age",
                        "total_rooms",
                        "total_bedrooms",
                        "population",
                        "households",
                        "median_income",
                    ],
                ),
                ("cat", OneHotEncoder(), ["ocean_proximity"]),
            ]
        )

        # Create a pipeline with preprocessing and the model
        model = Pipeline(
            steps=[
                ("preprocessor", preprocessor),
                ("regressor", RandomForestRegressor(random_state=42)),
            ]
        )

        # Log model parameters
        mlflow.log_param("n_estimators", model.named_steps["regressor"].n_estimators)
        mlflow.log_param("max_depth", model.named_steps["regressor"].max_depth)

        # Fit the model
        model.fit(X_train, y_train)

        # Validate model
        predictions = model.predict(X_valid)
        rmse = mean_squared_error(y_valid, predictions, squared=False)

        # Log metrics
        mlflow.log_metric("rmse", rmse)

        # Save the model and log it
        model_file_path = os.path.join(data_folder, "housing_model.pkl")
        joblib.dump(model, model_file_path)
        mlflow.sklearn.log_model(model, "model")

        print(f"Model saved to {model_file_path}")
