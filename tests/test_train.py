import os
import unittest
from unittest.mock import patch, MagicMock
import pandas as pd
from train import train_model


class TestTrainModel(unittest.TestCase):

    @patch("train.pd.read_csv")
    @patch("train.joblib.dump")
    @patch("train.train_test_split")
    def test_train_model(self, mock_train_test_split, mock_joblib_dump, mock_read_csv):
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
            "median_house_value": [200000, 300000],
        }
        mock_read_csv.return_value = pd.DataFrame(mock_data)

        # Mocking the output of train_test_split
        mock_train_test_split.return_value = (
            MagicMock(),
            MagicMock(),
            MagicMock(),
            MagicMock(),
        )

        # Define a temporary data folder
        temp_data_folder = "temp_data_folder"
        os.makedirs(temp_data_folder, exist_ok=True)

        try:
            # Call the function to test
            train_model(temp_data_folder)

            # Check that read_csv was called once
            mock_read_csv.assert_called_once_with(
                os.path.join(temp_data_folder, "housing_data.csv")
            )

            # Check that joblib.dump was called once with the model and correct file path
            mock_joblib_dump.assert_called_once()
            self.assertIn("housing_model.pkl", mock_joblib_dump.call_args[0][1])

        finally:
            # Clean up the temporary directory
            if os.path.exists(temp_data_folder):
                os.rmdir(temp_data_folder)


if __name__ == "__main__":
    unittest.main()
