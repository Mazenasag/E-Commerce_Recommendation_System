import numpy as np
import faiss
from scipy.sparse import csr_matrix
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from implicit.als import AlternatingLeastSquares
from tqdm import tqdm
from typing import Dict, Tuple
import sys
from src.utils.logger import get_logger
from src.utils.exception import CustomException

logger = get_logger(__name__)

class ModelTrainer:
    def __init__(self, config: dict):
        model_config = config.get('model', {})
        self.random_seed = model_config.get('random_seed', 42)
        np.random.seed(self.random_seed)
        
        als_config = model_config.get('als', {})
        self.als_factors = als_config.get('factors', 64)
        self.als_regularization = als_config.get('regularization', 0.3)
        self.als_iterations = als_config.get('iterations', 30)
        self.als_alpha = als_config.get('alpha', 40)
        
        emb_config = model_config.get('embeddings', {})
        self.max_features = emb_config.get('max_features', 100)
        self.ngram_range = tuple(emb_config.get('ngram_range', [1, 2]))
        self.min_df = emb_config.get('min_df', 3)
        self.n_components = emb_config.get('n_components', 50)
    
    def create_product_embeddings(self, product_info):
        try:
            product_info = product_info.drop_duplicates(subset=['product_id']).dropna(subset=['cleaned_text'])
            product_info['cleaned_text'] = product_info['cleaned_text'].fillna('').astype(str)
            logger.info(f"Creating embeddings for {len(product_info):,} products...")
            
            tfidf = TfidfVectorizer(
                max_features=self.max_features,
                ngram_range=self.ngram_range,
                min_df=self.min_df,
                stop_words=None,
                dtype=np.float32
            )
            
            product_embeddings_tfidf = tfidf.fit_transform(product_info['cleaned_text'])
            logger.info(f"TF-IDF shape: {product_embeddings_tfidf.shape}")
            
            n_features = product_embeddings_tfidf.shape[1]
            n_documents = product_embeddings_tfidf.shape[0]
            actual_n_components = min(self.n_components, n_features, n_documents)
            
            svd = TruncatedSVD(n_components=actual_n_components, random_state=self.random_seed)
            product_embeddings = svd.fit_transform(product_embeddings_tfidf).astype('float32')
            logger.info(f"Embeddings shape: {product_embeddings.shape}, Explained variance: {svd.explained_variance_ratio_.sum():.2%}")
            
            return product_embeddings, tfidf, svd, product_info
        except Exception as e:
            raise CustomException(f"Failed to create embeddings: {str(e)}", sys)
    
    def build_faiss_index(self, embeddings):
        try:
            faiss.normalize_L2(embeddings)
            index = faiss.IndexFlatL2(embeddings.shape[1])
            index.add(embeddings)
            logger.info(f"FAISS index built: {index.ntotal:,} vectors")
            return index
        except Exception as e:
            raise CustomException(f"Failed to build FAISS index: {str(e)}", sys)
    
    def train_als_model(self, train_matrix):
        try:
            logger.info(f"Training ALS model: {train_matrix.shape[0]:,} users, {train_matrix.shape[1]:,} products")
            als_model = AlternatingLeastSquares(
                factors=self.als_factors,
                regularization=self.als_regularization,
                iterations=self.als_iterations,
                random_state=self.random_seed
            )
            als_model.fit(train_matrix.T * self.als_alpha)
            logger.info("ALS model trained")
            return als_model
        except Exception as e:
            raise CustomException(f"Failed to train ALS model: {str(e)}", sys)
    
    def build_interaction_matrix(self, df, mappings, user_segment=None):
        try:
            if user_segment:
                user_id_to_idx = mappings['user_id_to_idx']
                user_indices = set([user_id_to_idx[uid] for uid in user_segment if uid in user_id_to_idx])
                df_filtered = df[df['user_idx'].isin(user_indices)].copy()
            else:
                df_filtered = df.copy()
            
            train_scores = df_filtered.groupby(['user_idx', 'product_idx'])['score'].sum().reset_index()
            n_users = mappings['n_users']
            n_products = mappings['n_products']
            
            if user_segment:
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
                matrix = csr_matrix(
                    (train_scores['score'].values,
                     (train_scores['user_idx'].values, train_scores['product_idx'].values)),
                    shape=(n_users, n_products)
                )
                warm_user_info = {}
            
            logger.info(f"Interaction matrix: {matrix.shape}, Non-zero: {matrix.nnz:,}")
            return matrix, warm_user_info
        except Exception as e:
            raise CustomException(f"Failed to build interaction matrix: {str(e)}", sys)
    
    def build_product_user_lookup(self, interaction_matrix, mappings, products_with_embeddings=None):
        try:
            if products_with_embeddings is None:
                products_with_embeddings = set(mappings['product_id_to_idx'].keys())
            
            idx_to_product_id = mappings['idx_to_product_id']
            coo_matrix = interaction_matrix.tocoo()
            product_to_users = {}
            
            for user_idx, product_idx, score in tqdm(
                zip(coo_matrix.row, coo_matrix.col, coo_matrix.data),
                total=len(coo_matrix.data),
                desc="Building lookup"
            ):
                orig_pid = idx_to_product_id.get(int(product_idx))
                if orig_pid is not None and orig_pid in products_with_embeddings:
                    if int(product_idx) not in product_to_users:
                        product_to_users[int(product_idx)] = []
                    product_to_users[int(product_idx)].append((int(user_idx), float(score)))
            
            logger.info(f"Product-user lookup built: {len(product_to_users):,} products")
            return dict(product_to_users)
        except Exception as e:
            raise CustomException(f"Failed to build product-user lookup: {str(e)}", sys)

