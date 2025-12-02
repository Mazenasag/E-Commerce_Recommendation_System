"""
Model Trainer Component
Trains ALS model and creates product embeddings
"""
import pandas as pd
import numpy as np
import pickle
import json
import faiss
from pathlib import Path
from scipy.sparse import csr_matrix
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from implicit.als import AlternatingLeastSquares
from collections import defaultdict
from tqdm import tqdm
from typing import Dict, List, Tuple
import sys
from src.utils.logger import get_logger
from src.utils.exception import CustomException
from typing import Set, Dict, Tuple


logger = get_logger(__name__)

class ModelTrainer:
    """Handles model training for ALS and embeddings"""
    
    def __init__(self, config: dict):
        """
        Initialize model trainer
        
        Args:
            config: Configuration dictionary
        """
        self.config = config
        self.model_config = config.get('model', {})
        self.random_seed = self.model_config.get('random_seed', 42)
        np.random.seed(self.random_seed)
        
        # ALS parameters
        als_config = self.model_config.get('als', {})
        self.als_factors = als_config.get('factors', 64)
        self.als_regularization = als_config.get('regularization', 0.3)
        self.als_iterations = als_config.get('iterations', 30)
        self.als_alpha = als_config.get('alpha', 40)
        
        # Embedding parameters
        emb_config = self.model_config.get('embeddings', {})
        self.max_features = emb_config.get('max_features', 100)
        self.ngram_range = tuple(emb_config.get('ngram_range', [1, 2]))
        self.min_df = emb_config.get('min_df', 3)
        self.n_components = emb_config.get('n_components', 50)
    
    def create_product_embeddings(self, product_info: pd.DataFrame) -> Tuple[np.ndarray, TfidfVectorizer, TruncatedSVD, pd.DataFrame]:
        """
        Create product embeddings using TF-IDF + SVD
        EXACTLY matches model_build.ipynb notebook implementation
        
        Args:
            product_info: DataFrame with product_id and cleaned_text columns
        
        Returns:
            Tuple of (embeddings, tfidf_vectorizer, svd_transformer, filtered_product_info)
        """
        try:
            logger.info("🔄 Creating product embeddings...")
            
            product_info = product_info.drop_duplicates(subset=['product_id'])
            product_info = product_info.dropna(subset=['cleaned_text'])
            
            logger.info(f"   📊 Products with cleaned text: {len(product_info):,}")
            
            product_info['cleaned_text'] = product_info['cleaned_text'].fillna('').astype(str)
            
            texts = product_info['cleaned_text']
            
            logger.info("   Creating TF-IDF vectors...")
            tfidf = TfidfVectorizer(
                max_features=self.max_features,
                ngram_range=self.ngram_range,
                min_df=self.min_df,
                stop_words=None,
                dtype=np.float32
            )
            
            product_embeddings_tfidf = tfidf.fit_transform(texts)
            logger.info(f"   ✅ TF-IDF shape: {product_embeddings_tfidf.shape} (sparse)")
            
            n_features = product_embeddings_tfidf.shape[1]
            n_documents = product_embeddings_tfidf.shape[0]
            
            actual_n_components = min(self.n_components, n_features, n_documents)
            if actual_n_components < self.n_components:
                logger.warning(
                    f"   ⚠️  Requested n_components ({self.n_components}) exceeds available features ({n_features}) or documents ({n_documents}). "
                    f"Using {actual_n_components} components instead."
                )
            
            logger.info(f"   Applying SVD with {actual_n_components} components...")
            svd = TruncatedSVD(n_components=actual_n_components, random_state=self.random_seed)
            product_embeddings = svd.fit_transform(product_embeddings_tfidf).astype('float32')
            
            explained_variance = svd.explained_variance_ratio_.sum()
            logger.info(f"   ✅ Reduced shape: {product_embeddings.shape}")
            logger.info(f"   ✅ Explained variance: {explained_variance:.2%}")
            
            return product_embeddings, tfidf, svd, product_info
            
        except Exception as e:
            logger.error(f"Error creating product embeddings: {e}")
            raise CustomException(f"Failed to create embeddings: {str(e)}", sys)
    
    def build_faiss_index(self, embeddings: np.ndarray) -> faiss.Index:
        """
        Build FAISS index for fast similarity search
        
        Args:
            embeddings: Product embeddings array
        
        Returns:
            FAISS index
        """
        try:
            logger.info("🔄 Building FAISS index...")
            
            dimension = embeddings.shape[1]
            
            # Normalize for cosine similarity
            faiss.normalize_L2(embeddings)
            
            # Create index
            index = faiss.IndexFlatL2(dimension)
            index.add(embeddings)
            
            logger.info(f"   ✅ FAISS Index: {index.ntotal:,} vectors, dimension {index.d}")
            return index
            
        except Exception as e:
            logger.error(f"Error building FAISS index: {e}")
            raise CustomException(f"Failed to build FAISS index: {str(e)}", sys)
    
    def train_als_model(self, train_matrix: csr_matrix) -> AlternatingLeastSquares:
        """
        Train ALS (Alternating Least Squares) model
        
        Args:
            train_matrix: Sparse user-product interaction matrix
        
        Returns:
            Trained ALS model
        """
        try:
            logger.info("🔄 Training ALS model...")
            
            als_model = AlternatingLeastSquares(
                factors=self.als_factors,
                regularization=self.als_regularization,
                iterations=self.als_iterations,
                random_state=self.random_seed
            )
            
            # Train on transposed matrix (items x users)
            als_model.fit(train_matrix.T * self.als_alpha)
            
            logger.info("✅ ALS model trained")
            return als_model
            
        except Exception as e:
            logger.error(f"Error training ALS model: {e}")
            raise CustomException(f"Failed to train ALS model: {str(e)}", sys)
    
    def build_interaction_matrix(self, df: pd.DataFrame, mappings: Dict, 
                                 user_segment: Set = None) -> Tuple[csr_matrix, Dict]:
        """
        Build sparse interaction matrix
        EXACTLY matches notebook: uses actual user_idx and product_idx values
        
        Args:
            df: DataFrame with user_idx, product_idx, score columns
            mappings: ID mappings dictionary
            user_segment: Set of user IDs to include (None = all users)
        
        Returns:
            Tuple of (sparse_matrix, warm_user_info)
        """
        try:
            logger.info("🔄 Building interaction matrix...")
            
            # Filter to user segment if provided
            if user_segment:
                user_id_to_idx = mappings['user_id_to_idx']
                user_indices = set([user_id_to_idx[uid] for uid in user_segment if uid in user_id_to_idx])
                df_filtered = df[df['user_idx'].isin(user_indices)].copy()
            else:
                df_filtered = df.copy()
            
            # Aggregate scores
            train_scores = df_filtered.groupby(['user_idx', 'product_idx'])['score'].sum().reset_index()
            
            n_users = mappings['n_users']
            n_products = mappings['n_products']
            
            if user_segment:
                # For warm users: use matrix indices (for ALS training)
                user_list = sorted(df_filtered['user_idx'].unique())
                user_to_matrix_idx = {uid: i for i, uid in enumerate(user_list)}
                
                matrix = csr_matrix(
                    (train_scores['score'].values,
                     ([user_to_matrix_idx[u] for u in train_scores['user_idx']], 
                      train_scores['product_idx'].values)),
                    shape=(len(user_list), n_products)
                )
                
                warm_user_info = {
                    'warm_user_list': user_list,
                    'warm_user_to_matrix_idx': user_to_matrix_idx
                }
            else:
                # For full interaction matrix: use actual user_idx values (EXACTLY like notebook line 2755)
                # Notebook: interaction_matrix = csr_matrix((train_scores['score'].values, (train_scores['user_idx'].values, train_scores['product_idx'].values)), shape=(n_users, n_products))
                matrix = csr_matrix(
                    (train_scores['score'].values,
                     (train_scores['user_idx'].values, train_scores['product_idx'].values)),
                    shape=(n_users, n_products)
                )
                
                warm_user_info = {}
            
            logger.info(f"   ✅ Matrix shape: {matrix.shape}, Non-zero: {matrix.nnz:,}")
            
            return matrix, warm_user_info
            
        except Exception as e:
            logger.error(f"Error building interaction matrix: {e}")
            raise CustomException(f"Failed to build interaction matrix: {str(e)}", sys)
    
    def build_product_user_lookup(self, interaction_matrix: csr_matrix, 
                                   mappings: Dict, products_with_embeddings: set = None) -> Dict:
        """
        Build product-to-users lookup for content-based recommendations
        EXACTLY matches notebook: only includes products with embeddings
        
        Args:
            interaction_matrix: Sparse interaction matrix
            mappings: ID mappings dictionary
            products_with_embeddings: Set of product IDs that have embeddings (from product_info)
        
        Returns:
            Dictionary mapping product_idx to list of (user_idx, score) tuples
        """
        try:
            logger.info("🔄 Building product-user lookup...")
            
            if products_with_embeddings is None:
                products_with_embeddings = set(mappings['product_id_to_idx'].keys())
            
            idx_to_product_id = mappings['idx_to_product_id']
            
            coo_matrix = interaction_matrix.tocoo()
            
            # EXACTLY like notebook: product_to_users_list = {}  # {product_idx: [(user_idx, score), ...]}
            product_to_users = {}
            
            # EXACTLY like notebook: iterate through COO matrix entries
            # Notebook: for user_idx, product_idx, score in tqdm(zip(coo_matrix.row, coo_matrix.col, coo_matrix.data), ...)
            for user_idx, product_idx, score in tqdm(
                zip(coo_matrix.row, coo_matrix.col, coo_matrix.data),
                total=len(coo_matrix.data),
                desc="Building lookup"
            ):
                # EXACTLY like notebook line 2782-2786:
                # orig_pid = idx_to_product_id.get(product_idx)
                # if orig_pid and orig_pid in products_with_embeddings:
                #     if product_idx not in product_to_users_list:
                #         product_to_users_list[product_idx] = []
                #     product_to_users_list[product_idx].append((int(user_idx), float(score)))
                orig_pid = idx_to_product_id.get(int(product_idx))
                if orig_pid is not None and orig_pid in products_with_embeddings:
                    if int(product_idx) not in product_to_users:
                        product_to_users[int(product_idx)] = []
                    product_to_users[int(product_idx)].append((int(user_idx), float(score)))
            
            logger.info(f"   ✅ Lookup built for {len(product_to_users):,} products")
            return dict(product_to_users)
            
        except Exception as e:
            logger.error(f"Error building product-user lookup: {e}")
            raise CustomException(f"Failed to build product-user lookup: {str(e)}", sys)

