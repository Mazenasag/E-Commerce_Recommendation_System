
# Data Ingestion Component Loads raw data from CSV files

import sys
import pandas as pd
from pathlib import Path
from src.utils.logger import get_logger
from src.utils.exception import CustomException

logger = get_logger(__name__)

class DataIngestion:
    """Handles data loading from various sources"""

    def __init__(self, config: dict):
        self.config = config
        self.raw_data_path = config.get('data', {}).get(
            'raw_data_path', 'data/raw/csv_for_case_study_V1.csv'
        )

    def load_data(self) -> pd.DataFrame:
        """Load raw data from CSV file"""
        try:
            logger.info(f"Loading data from: {self.raw_data_path}")
            
            data_path = Path(self.raw_data_path)
            if not data_path.exists():
                raise CustomException(f"Data file not found: {self.raw_data_path}", sys)

            df = pd.read_csv(self.raw_data_path)
            
            logger.info(f" Data loaded: {len(df):,} rows, {len(df.columns)} columns")
            logger.info(f"Columns: {list(df.columns)}")
            return df

        except Exception as e:
            raise CustomException(f"Failed to load data: {e}", sys)

    def validate_data(self, df: pd.DataFrame) -> bool:
        """Validate that required columns exist"""
        try:
            required_columns = ['product_id', 'customer_id', 'product_name', 'Event_Date', 'Event']
            missing_columns = [col for col in required_columns if col not in df.columns]
            if missing_columns:
                raise CustomException(f"Missing columns: {missing_columns}", sys)

            logger.info(" Data validation passed")
            return True

        except Exception as e:
            raise CustomException(f"Data validation failed: {e}", sys)
if __name__ == "__main__":
    import yaml

    # Load config (optional: default path)
    config_path = "config/config.yaml"
    try:
        with open(config_path, "r") as f:
            config = yaml.safe_load(f)
    except Exception:
        config = {"data": {"raw_data_path": "data/raw/csv_for_case_study_V1.csv"}}

    # Create DataIngestion instance
    ingestion = DataIngestion(config)

    # Load and validate data
    try:
        df = ingestion.load_data()
        ingestion.validate_data(df)
        print(" Data loaded and validated successfully")
    except Exception as e:
        print(f" Error: {e}")
