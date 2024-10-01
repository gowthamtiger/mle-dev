import os
import unittest
from unittest.mock import patch, MagicMock
import pandas as pd
from score import score_model


class TestScoreModel(unittest.TestCase):

    @patch("score.joblib.load")
    @patch("score.pd.read_csv")
    @patch("score.pd.DataFrame.to_csv")
    def test_score_model(self, mock_to_csv, mock_read_csv, mock_joblib_load):
        # Mocking the model and its pipeline
        mock_model = MagicMock()
        mock_model.named_steps = {"preprocessor": MagicMock()}
        mock_model.predict.return_value = [200000, 300000]  # Mock predictions
        mock_joblib_load.return_value = mock_model

        # Mocking the input DataFrame
        mock_data = {
            "longitude": [1.0, 2.0],
            "latitude": [3.0, 4.0],
            "housing_median_age": [5, 6],
            "total_rooms": [7, 8],
            "total_bedrooms": [9, 10],
            "population": [11, 12],
            "households": [13, 14],
            "median_income": [15, 16],
            "ocean_proximity": ["NEAR BAY", "ISLAND"],
        }
        mock_read_csv.return_value = pd.DataFrame(mock_data)

        # Define temporary model and data folders
        temp_model_folder = "temp_model_folder"
        temp_data_folder = "temp_data_folder"
        os.makedirs(temp_model_folder, exist_ok=True)
        os.makedirs(temp_data_folder, exist_ok=True)

        try:
            # Call the function to test
            score_model(temp_model_folder, temp_data_folder)

            # Check that the model was loaded correctly
            mock_joblib_load.assert_called_once_with(
                os.path.join(temp_model_folder, "housing_model.pkl")
            )

            # Check that read_csv was called once for the test data
            mock_read_csv.assert_called_once_with(
                os.path.join(temp_data_folder, "housing_data.csv")
            )

            # Check that the prediction results were saved to a CSV file
            mock_to_csv.assert_called_once()
            self.assertIn("predictions.csv", mock_to_csv.call_args[0][0])

            # Check the expected output DataFrame contains predictions
            output_df = mock_to_csv.call_args[0][0]
            self.assertIn("predicted_median_house_value", output_df.columns)

        finally:
            # Clean up the temporary directories
            if os.path.exists(temp_model_folder):
                os.rmdir(temp_model_folder)
            if os.path.exists(temp_data_folder):
                os.rmdir(temp_data_folder)


if __name__ == "__main__":
    unittest.main()
