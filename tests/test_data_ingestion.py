"""
Unit tests for data ingestion component
Run this file directly: python tests/test_data_ingestion.py
"""

import os
import sys
import pandas as pd
from pathlib import Path

# Add parent directory to PYTHONPATH
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.components.data_ingestion import DataIngestion
from src.utils.exception import CustomException


def test_data_ingestion_initialization():
    config = {"data": {"raw_data_path": "data/raw/csv_for_case_study_V1.csv"}}
    di = DataIngestion(config)
    assert di.config == config
    assert di.raw_data_path is not None


def test_load_data_missing_file():
    config = {"data": {"raw_data_path": "this_file_does_not_exist.csv"}}
    di = DataIngestion(config)

    try:
        di.load_data()
        print("Test failed: load_data() should raise exception for missing file")
    except CustomException:
        print("Test passed: missing file raised exception")


def test_validate_data_valid():
    config = {"data": {"raw_data_path": "data/raw/csv_for_case_study_V1.csv"}}
    di = DataIngestion(config)

    valid_df = pd.DataFrame({
        "product_id": [1, 2],
        "customer_id": [1, 2],
        "product_name": ["A", "B"],
        "Event_Date": ["2023-01-01", "2023-01-02"],
        "Event": ["purchased", "cart"]
    })

    result = di.validate_data(valid_df)
    assert result is True
    print("Test passed: validate_data() accepted valid data")


def test_validate_data_invalid():
    config = {"data": {"raw_data_path": "data/raw/csv_for_case_study_V1.csv"}}
    di = DataIngestion(config)

    invalid_df = pd.DataFrame({
        "product_id": [1, 2],
        "customer_id": [1, 2]
    })

    try:
        di.validate_data(invalid_df)
        print("Test failed: validate_data() should raise exception for missing columns")
    except CustomException:
        print("Test passed: missing columns raised exception")


if __name__ == "__main__":
    print("Running data ingestion tests...\n")

    test_data_ingestion_initialization()
    test_load_data_missing_file()
    test_validate_data_valid()
    test_validate_data_invalid()

    print("\nAll tests completed.")
