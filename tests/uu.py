"""
Unit test to call the full data transformation pipeline
and check that processed data is saved.
"""

import sys
import os
from pathlib import Path
import pytest

# Add src to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.components.data_ingestion import DataIngestion
from src.components.data_transformation import DataTransformation

@pytest.fixture
def test_config(tmp_path):
    """Create temporary config with a small CSV file"""
    # Create a small CSV file
    raw_csv = tmp_path / "test_raw.csv"
    raw_csv.write_text(
        "product_id,customer_id,product_name,Event_Date,Event\n"
        "101,1,Product A ML,2023-01-01,Purchased\n"
        "102,2,منتج B كيلو,,Cart\n"
        "103,1,Product C 250ml,2023-01-03,Wishlist\n"
    )

    processed_path = tmp_path / "processed_data.parquet"

    config = {
        "data": {
            "raw_data_path": str(raw_csv),
            "processed_data_path": str(processed_path),
        },
        "event_weights": {"purchased": 5.0, "cart": 3.0, "wishlist": 1.0},
        "recency": {"decay_rate": 0.01},
        "user_segmentation": {"warm_user_threshold": 2},
    }
    return config

def test_full_pipeline_runs(test_config):
    """Run the full pipeline and check that processed data is saved"""
    config = test_config

    # 1️⃣ Data Ingestion
    ingestion = DataIngestion(config)
    df = ingestion.load_data()
    ingestion.validate_data(df)

    # 2️⃣ Data Transformation
    transformer = DataTransformation(config)
    df = transformer.clean_data(df)
    df = transformer.process_product_names(df)
    df = transformer.create_weighted_scores(df)
    mappings = transformer.create_id_mappings(df)
    warm_users, cold_users = transformer.identify_user_segments(df)

    # 3️⃣ Save processed data
    processed_path = config["data"]["processed_data_path"]
    transformer.save_processed_data(df, processed_path)

    # -------------------------
    # Assertions
    # -------------------------
    assert Path(processed_path).exists(), "Processed file was not created"
    assert len(df) > 0, "Transformed DataFrame is empty"
    assert "score" in df.columns, "Weighted scores not calculated"
    assert len(warm_users) > 0, "No warm users identified"
    assert len(cold_users) > 0, "No cold users identified"
