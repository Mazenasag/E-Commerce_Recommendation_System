"""
Configuration loader for YAML/JSON configs
"""
import sys
import yaml
import json
from pathlib import Path
from typing import Dict, Any
from src.utils.logger import get_logger
from src.utils.exception import CustomException

logger = get_logger(__name__)

def load_config(config_path: str = "config/config.yaml") -> Dict[str, Any]:
    """
    Load configuration from YAML file
    
    Args:
        config_path: Path to config YAML file
    
    Returns:
        Configuration dictionary
    """
    config_path = Path(config_path)
    
    if not config_path.exists():
        logger.warning(f"Config file not found at {config_path}. Using defaults.")
        return get_default_config()
    
    try:
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
            logger.info(f"Configuration loaded from {config_path}")
            return config
    except Exception as e:
        logger.error(f"Error loading config from {config_path}: {e}")
        logger.info("Using default configuration")
        return get_default_config()

def get_default_config() -> Dict[str, Any]:
    """Get default configuration"""
    return {
        "data": {
            "raw_data_path": "data/raw/csv_for_case_study_V1.csv",
            "processed_data_path": "data/processed/processed_data.parquet",
            "train_test_split_ratio": 0.8
        },
        "model": {
            "random_seed": 42,
            "als": {
                "factors": 64,
                "regularization": 0.3,
                "iterations": 30,
                "alpha": 40
            },
            "embeddings": {
                "max_features": 100,
                "ngram_range": [1, 2],
                "min_df": 3,
                "n_components": 50
            }
        },
        "event_weights": {
            "purchased": 5.0,
            "cart": 3.0,
            "rating": 2.5,
            "wishlist": 2.0,
            "search_keyword": 1.0
        },
        "artifacts": {
            "base_path": "artifacts"
        }
    }

def save_config(config: Dict[str, Any], output_path: str):
    """Save configuration to JSON file"""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'w') as f:
        json.dump(config, f, indent=2)
    logger.info(f"Configuration saved to {output_path}")

