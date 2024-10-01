import unittest
from unittest.mock import patch, MagicMock
import os
import pandas as pd
import ingest_data  # Assuming your script is named ingest_data.py


class TestIngestData(unittest.TestCase):

    @patch("ingest_data.urllib.request.urlretrieve")
    @patch("ingest_data.tarfile.open")
    @patch("os.makedirs")
    def test_fetch_housing_data(
        self, mock_makedirs, mock_tarfile_open, mock_urlretrieve
    ):
        # Setup
        mock_urlretrieve.return_value = None  # Simulate successful download
        mock_tarfile = MagicMock()
        mock_tarfile_open.return_value.__enter__.return_value = mock_tarfile

        # Call the function
        ingest_data.fetch_housing_data()

        # Assertions
        mock_makedirs.assert_called_once_with(ingest_data.HOUSING_PATH, exist_ok=True)
        mock_urlretrieve.assert_called_once_with(
            ingest_data.HOUSING_URL,
            os.path.join(ingest_data.HOUSING_PATH, "housing.tgz"),
        )
        mock_tarfile.extractall.assert_called_once_with(path=ingest_data.HOUSING_PATH)
        mock_tarfile.close.assert_called_once()

    @patch("pandas.read_csv")
    def test_load_housing_data(self, mock_read_csv):
        # Setup
        mock_dataframe = pd.DataFrame({"col1": [1, 2], "col2": [3, 4]})
        mock_read_csv.return_value = mock_dataframe

        # Call the function
        result = ingest_data.load_housing_data()

        # Assertions
        mock_read_csv.assert_called_once_with(
            os.path.join(ingest_data.HOUSING_PATH, "housing.csv")
        )
        pd.testing.assert_frame_equal(result, mock_dataframe)

    @patch("ingest_data.fetch_housing_data")
    @patch("ingest_data.load_housing_data")
    @patch("pandas.DataFrame.to_csv")
    def test_main(self, mock_to_csv, mock_load_housing_data, mock_fetch_housing_data):
        # Setup
        mock_load_housing_data.return_value = pd.DataFrame(
            {"col1": [1, 2], "col2": [3, 4]}
        )
        output_folder = "output_test"

        # Call the main function
        ingest_data.main(output_folder)

        # Assertions
        mock_fetch_housing_data.assert_called_once()
        mock_load_housing_data.assert_called_once()
        mock_to_csv.assert_called_once_with(
            os.path.join(output_folder, "housing_data.csv"), index=False
        )

    @patch("os.path.join", return_value="mocked_path/housing.csv")
    @patch("pandas.read_csv", side_effect=FileNotFoundError)
    def test_load_housing_data_file_not_found(self, mock_read_csv, mock_join):
        # Call the function and assert it raises FileNotFoundError
        with self.assertRaises(FileNotFoundError):
            ingest_data.load_housing_data()


if __name__ == "__main__":
    unittest.main()
