import pandas as pd
import numpy as np
import re
from pathlib import Path
from typing import Dict, Set, Tuple
from sklearn.preprocessing import LabelEncoder
import sys
from src.utils.logger import get_logger
from src.utils.helpers import ensure_dir
from src.utils.exception import CustomException

logger = get_logger(__name__)

class DataTransformation:
    def __init__(self, config: dict):
        self.event_weights = config.get('event_weights', {})
        self.recency_decay = config.get('recency', {}).get('decay_rate', 0.01)
        self.warm_user_threshold = config.get('user_segmentation', {}).get('warm_user_threshold', 2)
        self.arabic_stopwords = {
            "من", "مع", "في", "على", "و", "الى", "عن", "هذا", "ذلك", 
            "او", "اي", "كل", "ثم", "هو", "هي"
        }
    
    def clean_data(self, df):
        try:
            df.columns = df.columns.str.lower().str.strip().str.replace(' ', '_')
            df['event'] = df['event'].str.lower().str.strip()
            df['event_date'] = pd.to_datetime(df['event_date'], errors='coerce')
            df['event_date'] = df['event_date'].fillna(df['event_date'].median())
            if 'index' in df.columns:
                df = df.drop('index', axis=1)
            return df
        except Exception as e:
            raise CustomException(f"Failed to clean data: {str(e)}", sys)
    
    def normalize_units(self, text: str) -> str:
        """Normalize unit variations to standard form"""
        text = str(text)
        # Normalize ML / مل
        ml_patterns = [r"\bML\b", r"\bMl\b", r"\bml\b", r"\bمل\b", r"\bملي\b", r"\bمليلتر\b"]
        for pat in ml_patterns:
            text = re.sub(pat, " مل ", text, flags=re.IGNORECASE)
        # Normalize KG / كيلو
        kg_patterns = [r"\bKG\b", r"\bKg\b", r"\bkg\b", r"\bكيلو\b", r"\bكغ\b"]
        for pat in kg_patterns:
            text = re.sub(pat, " كيلو ", text, flags=re.IGNORECASE)
        return text
    
    def clean_product_name(self, name):
        name = str(name)
        name = self.normalize_units(name)
        name = re.sub(r"[^\w\s\u0600-\u06FF]", " ", name)
        name = re.sub(r"\s+", " ", name).strip()
        words = name.split()
        words = [w for w in words if w not in self.arabic_stopwords]
        return words
    
    def process_product_names(self, df):
        try:
            df['clean_words'] = df['product_name'].astype(str).apply(self.clean_product_name)
            df['cleaned_text'] = df['clean_words'].apply(lambda x: " ".join(x))
            return df
        except Exception as e:
            raise CustomException(f"Failed to process product names: {str(e)}", sys)
    
    def create_weighted_scores(self, df):
        try:
            df['event_weight'] = df['event'].map(self.event_weights).fillna(1.0)
            reference_date = df['event_date'].max()
            df['days_ago'] = (reference_date - df['event_date']).dt.days
            df['recency_weight'] = np.exp(-self.recency_decay * df['days_ago'])
            df['score'] = df['event_weight'] * df['recency_weight']
            return df
        except Exception as e:
            raise CustomException(f"Failed to create weighted scores: {str(e)}", sys)
    
    def create_id_mappings(self, df):
        try:
            user_encoder = LabelEncoder()
            product_encoder = LabelEncoder()
            df['user_idx'] = user_encoder.fit_transform(df['customer_id'])
            df['product_idx'] = product_encoder.fit_transform(df['product_id'])
            mappings = {
                'user_id_to_idx': dict(zip(df['customer_id'], df['user_idx'])),
                'idx_to_user_id': dict(zip(df['user_idx'], df['customer_id'])),
                'product_id_to_idx': dict(zip(df['product_id'], df['product_idx'])),
                'idx_to_product_id': dict(zip(df['product_idx'], df['product_id'])),
                'n_users': df['user_idx'].nunique(),
                'n_products': df['product_idx'].nunique()
            }
            logger.info(f"ID mappings: {mappings['n_users']:,} users, {mappings['n_products']:,} products")
            return mappings
        except Exception as e:
            raise CustomException(f"Failed to create ID mappings: {str(e)}", sys)
    
    def identify_user_segments(self, df):
        try:
            user_counts = df.groupby('customer_id').size()
            warm_users = set(user_counts[user_counts >= self.warm_user_threshold].index)
            cold_users = set(user_counts[user_counts < self.warm_user_threshold].index)
            logger.info(f"User segments: {len(warm_users):,} warm, {len(cold_users):,} cold")
            return warm_users, cold_users
        except Exception as e:
            raise CustomException(f"Failed to identify user segments: {str(e)}", sys)
    
    def save_processed_data(self, df, output_path):
        try:
            ensure_dir(Path(output_path).parent)
            if output_path.endswith('.parquet'):
                csv_path = str(Path(output_path).with_suffix('.csv'))
            elif output_path.endswith('.csv'):
                csv_path = output_path
            else:
                csv_path = output_path + '.csv'
            df.to_csv(csv_path, index=False)
        except Exception as e:
            logger.warning(f"Could not save processed data: {e}")
    
    def load_processed_data(self, input_path):
        try:
            if input_path.endswith('.parquet'):
                csv_path = input_path.replace('.parquet', '.csv')
            elif input_path.endswith('.csv'):
                csv_path = input_path
            else:
                csv_path = input_path + '.csv'
            
            df = pd.read_csv(csv_path)
            
            if 'event_date' in df.columns:
                df['event_date'] = pd.to_datetime(df['event_date'])
            if 'user_idx' in df.columns:
                df['user_idx'] = df['user_idx'].astype('int64')
            if 'product_idx' in df.columns:
                df['product_idx'] = df['product_idx'].astype('int64')
            if 'score' in df.columns:
                df['score'] = df['score'].astype('float64')
            
            return df
        except Exception as e:
            raise CustomException(f"Could not load processed data: {e}", sys)

if __name__ == "__main__":
    import yaml
    import sys
    from src.components.data_ingestion import DataIngestion

    # Load config
    config_path = "config/config.yaml"
    try:
        with open(config_path, "r") as f:
            config = yaml.safe_load(f)
    except Exception:
        # Default config if YAML not found
        config = {
            "data": {"raw_data_path": "data/raw/csv_for_case_study_V1.csv"},
            "event_weights": {"view": 1.0, "purchase": 3.0},
            "recency": {"decay_rate": 0.01},
            "user_segmentation": {"warm_user_threshold": 2}
        }

    # Step 1: Load raw data
    ingestion = DataIngestion(config)
    try:
        df = ingestion.load_data()
        ingestion.validate_data(df)
    except Exception as e:
        print(f"❌ Data ingestion failed: {e}")
        sys.exit(1)

    # Step 2: Transform data
    transformer = DataTransformation(config)
    try:
        df = transformer.clean_data(df)
        df = transformer.process_product_names(df)
        df = transformer.create_weighted_scores(df)
        mappings = transformer.create_id_mappings(df)
        warm_users, cold_users = transformer.identify_user_segments(df)
        transformer.save_processed_data(df, "data/processed/processed_data.parquet")
        print("✅ Data transformation completed successfully")
    except Exception as e:
        print(f"❌ Data transformation failed: {e}")
        sys.exit(1)