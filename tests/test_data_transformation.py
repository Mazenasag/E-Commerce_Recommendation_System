"""
Unit tests for data transformation component
Run this file directly: python tests/test_data_transformation.py
"""
import os
import sys
import pandas as pd

# Add parent directory to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.components.data_transformation import DataTransformation
from src.utils.exception import CustomException

def test_data_transformation_initialization():
    """Test DataTransformation initialization"""
    config = {
        "event_weights": {"purchased": 5.0, "cart": 3.0},
        "recency": {"decay_rate": 0.01},
        "user_segmentation": {"warm_user_threshold": 2}
    }
    dt = DataTransformation(config)
    assert dt.config == config
    assert dt.event_weights is not None

def test_clean_data():
    """Test data cleaning"""
    config = {"event_weights": {}, "recency": {}, "user_segmentation": {}}
    dt = DataTransformation(config)
    
    df = pd.DataFrame({
        'Product_ID': [1, 2],
        'Customer_ID': [1, 2],
        'Product_Name': ['A', 'B'],
        'Event_Date': ['2023-01-01', None],
        'Event': ['Purchased', 'Cart']
    })
    
    cleaned = dt.clean_data(df)
    assert 'product_id' in cleaned.columns
    assert 'customer_id' in cleaned.columns

def test_create_weighted_scores():
    """Test weighted score creation"""
    config = {
        "event_weights": {"purchased": 5.0, "cart": 3.0},
        "recency": {"decay_rate": 0.01},
        "user_segmentation": {}
    }
    dt = DataTransformation(config)
    
    df = pd.DataFrame({
        'event': ['purchased', 'cart', 'wishlist'],
        'event_date': pd.to_datetime(['2023-01-01', '2023-01-02', '2023-01-03'])
    })
    
    df = dt.create_weighted_scores(df)
    assert 'score' in df.columns
    assert df['score'].min() > 0

if __name__ == "__main__":
    test_data_transformation_initialization()
    print("✅ Data transformation initialization test passed!")
    
    test_clean_data()
    print("✅ Data cleaning test passed!")
    
    test_create_weighted_scores()
    print("✅ Weighted scores test passed!")
    
    print("\n✅ All data transformation tests passed!")
