"""
Data Transformation Component Handles data cleaning, preprocessing, and feature engineering
"""
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
    """Handles data transformation and preprocessing"""
    
    def __init__(self, config: dict):
        """
        Initialize data transformation
        
        Args:
            config: Configuration dictionary
        """
        self.config = config
        self.event_weights = config.get('event_weights', {})
        self.recency_decay = config.get('recency', {}).get('decay_rate', 0.01)
        self.warm_user_threshold = config.get('user_segmentation', {}).get('warm_user_threshold', 2)
        
        # Arabic stopwords
        self.arabic_stopwords = {
            "من", "مع", "في", "على", "و", "الى", "عن", "هذا", "ذلك", 
            "او", "اي", "كل", "ثم", "هو", "هي"
        }
    
    def clean_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Clean and normalize data
        
        Args:
            df: Raw DataFrame
        
        Returns:
            Cleaned DataFrame
        """
        try:
            logger.info("🔄 Cleaning data...")
            
            # Normalize column names
            df.columns = df.columns.str.lower().str.strip().str.replace(' ', '_')
            
            # Clean event column
            df['event'] = df['event'].str.lower().str.strip()
            
            # Handle dates
            df['event_date'] = pd.to_datetime(df['event_date'], errors='coerce')
            median_date = df['event_date'].median()
            df['event_date'] = df['event_date'].fillna(median_date)
            
            # Remove index column if exists
            if 'index' in df.columns:
                df = df.drop('index', axis=1)
            
            logger.info(f"✅ Data cleaned: {len(df):,} rows")
            return df
            
        except Exception as e:
            logger.error(f"Error cleaning data: {e}")
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
    
    def clean_product_name(self, name: str) -> list:
        """Clean product name and return word list - EXACTLY matches notebook normalize_and_clean_simple"""
        name = str(name)
        # Apply unit normalization
        name = self.normalize_units(name)
        # Keep only Arabic, English letters, and spaces (NOT removing numbers - matches notebook)
        name = re.sub(r"[^\w\s\u0600-\u06FF]", " ", name)
        # Normalize multiple spaces
        name = re.sub(r"\s+", " ", name).strip()
        # Tokenize
        words = name.split()
        # Remove stopwords
        words = [w for w in words if w not in self.arabic_stopwords]
        return words
    
    def process_product_names(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Process and clean product names
        
        Args:
            df: DataFrame with product_name column
        
        Returns:
            DataFrame with cleaned_text and clean_words columns
        """
        try:
            logger.info("🔄 Processing product names...")
            
            df['clean_words'] = df['product_name'].astype(str).apply(self.clean_product_name)
            df['cleaned_text'] = df['clean_words'].apply(lambda x: " ".join(x))
            
            logger.info("✅ Product names processed")
            return df
            
        except Exception as e:
            logger.error(f"Error processing product names: {e}")
            raise CustomException(f"Failed to process product names: {str(e)}", sys)
    
    def create_weighted_scores(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create weighted scores based on event type and recency
        
        Args:
            df: DataFrame with event and event_date columns
        
        Returns:
            DataFrame with score column
        """
        try:
            logger.info("🔄 Creating weighted scores...")
            
            # Event weights
            df['event_weight'] = df['event'].map(self.event_weights).fillna(1.0)
            
            # Recency weight
            reference_date = df['event_date'].max()
            df['days_ago'] = (reference_date - df['event_date']).dt.days
            df['recency_weight'] = np.exp(-self.recency_decay * df['days_ago'])
            
            # Combined score
            df['score'] = df['event_weight'] * df['recency_weight']
            
            logger.info("✅ Weighted scores created")
            return df
            
        except Exception as e:
            logger.error(f"Error creating weighted scores: {e}")
            raise CustomException(f"Failed to create weighted scores: {str(e)}", sys)
    
    def create_id_mappings(self, df: pd.DataFrame) -> Dict:
        """
        Create ID mappings for users and products
        
        Args:
            df: DataFrame with customer_id and product_id columns
        
        Returns:
            Dictionary with all mappings
        """
        try:
            logger.info("🔄 Creating ID mappings...")
            
            # Create encoders
            user_encoder = LabelEncoder()
            product_encoder = LabelEncoder()
            
            # Fit and transform
            df['user_idx'] = user_encoder.fit_transform(df['customer_id'])
            df['product_idx'] = product_encoder.fit_transform(df['product_id'])
            
            # Create mappings
            mappings = {
                'user_id_to_idx': dict(zip(df['customer_id'], df['user_idx'])),
                'idx_to_user_id': dict(zip(df['user_idx'], df['customer_id'])),
                'product_id_to_idx': dict(zip(df['product_id'], df['product_idx'])),
                'idx_to_product_id': dict(zip(df['product_idx'], df['product_id'])),
                'n_users': df['user_idx'].nunique(),
                'n_products': df['product_idx'].nunique()
            }
            
            logger.info(f"✅ Mappings created: {mappings['n_users']:,} users, {mappings['n_products']:,} products")
            return mappings
            
        except Exception as e:
            logger.error(f"Error creating ID mappings: {e}")
            raise CustomException(f"Failed to create ID mappings: {str(e)}", sys)
    
    def identify_user_segments(self, df: pd.DataFrame) -> Tuple[Set, Set]:
        """
        Identify warm and cold users
        
        Args:
            df: DataFrame with customer_id column
        
        Returns:
            Tuple of (warm_users, cold_users) sets
        """
        try:
            logger.info("🔄 Identifying user segments...")
            
            user_counts = df.groupby('customer_id').size()
            warm_users = set(user_counts[user_counts >= self.warm_user_threshold].index)
            cold_users = set(user_counts[user_counts < self.warm_user_threshold].index)
            
            logger.info(f"✅ User segments: {len(warm_users):,} warm, {len(cold_users):,} cold")
            return warm_users, cold_users
            
        except Exception as e:
            logger.error(f"Error identifying user segments: {e}")
            raise CustomException(f"Failed to identify user segments: {str(e)}", sys)
    
    def save_processed_data(self, df: pd.DataFrame, output_path: str):
        """
        Save processed data to disk (CSV format only)
        
        Args:
            df: Processed DataFrame
            output_path: Path to save processed data (will be converted to .csv)
        """
        try:
            ensure_dir(Path(output_path).parent)
            
            # Always save as CSV, convert any .parquet extension to .csv
            if output_path.endswith('.parquet'):
                csv_path = str(Path(output_path).with_suffix('.csv'))
            elif output_path.endswith('.csv'):
                csv_path = output_path
            else:
                csv_path = output_path if output_path.endswith('.csv') else output_path + '.csv'
            
            df.to_csv(csv_path, index=False)
            logger.info(f"✅ Processed data saved to {csv_path} ({len(df):,} rows)")
            
            # Remove old parquet file if it exists
            parquet_path = Path(output_path)
            if parquet_path.exists() and parquet_path.suffix == '.parquet':
                try:
                    parquet_path.unlink()
                    logger.info(f"   Removed old parquet file: {parquet_path}")
                except Exception:
                    pass
        except Exception as e:
            logger.warning(f"Could not save processed data: {e}")
    
    def build_product_user_lookup(self, df: pd.DataFrame, mappings: Dict) -> Dict:
        """
        Build product-to-users lookup from interaction data
        Created during preprocessing phase
        
        Args:
            df: DataFrame with user_idx, product_idx, score columns
            mappings: ID mappings dictionary
            
        Returns:
            Dictionary mapping product_idx to list of (user_idx, score) tuples
        """
        try:
            logger.info("🔄 Building product-user lookup from interactions...")
            
            # Group by product and user, sum scores
            product_user_scores = df.groupby(['product_idx', 'user_idx'])['score'].sum().reset_index()
            
            # Build lookup dictionary
            product_to_users = {}
            for _, row in product_user_scores.iterrows():
                product_idx = int(row['product_idx'])
                user_idx = int(row['user_idx'])
                score = float(row['score'])
                
                if product_idx not in product_to_users:
                    product_to_users[product_idx] = []
                product_to_users[product_idx].append((user_idx, score))
            
            logger.info(f"   ✅ Lookup built for {len(product_to_users):,} products")
            return product_to_users
            
        except Exception as e:
            logger.error(f"Error building product-user lookup: {e}")
            raise CustomException(f"Failed to build product-user lookup: {str(e)}", sys)
    
    def load_processed_data(self, input_path: str) -> pd.DataFrame:
        """
        Load processed data from disk (CSV format)
        
        Args:
            input_path: Path to load processed data from
            
        Returns:
            Loaded DataFrame with proper data types
        """
        try:
            if input_path.endswith('.parquet'):
                csv_path = input_path.replace('.parquet', '.csv')
            elif input_path.endswith('.csv'):
                csv_path = input_path
            else:
                csv_path = input_path + '.csv'
            
            if not Path(csv_path).exists():
                raise FileNotFoundError(f"Processed data file not found: {csv_path}")
            
            df = pd.read_csv(csv_path)
            
            # Convert data types
            if 'event_date' in df.columns:
                df['event_date'] = pd.to_datetime(df['event_date'])
            if 'user_idx' in df.columns:
                df['user_idx'] = df['user_idx'].astype('int64')
            if 'product_idx' in df.columns:
                df['product_idx'] = df['product_idx'].astype('int64')
            if 'score' in df.columns:
                df['score'] = df['score'].astype('float64')
            
            logger.info(f"✅ Processed data loaded from {csv_path} ({len(df):,} rows)")
            return df
        except Exception as e:
            logger.error(f"Could not load processed data: {e}")
            raise

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