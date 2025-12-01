"""
Prediction Component
Handles inference and recommendations
"""
import numpy as np
from typing import List, Dict, Optional
from collections import defaultdict
from scipy.sparse import csr_matrix
import sys
from src.utils.logger import get_logger
from src.utils.exception import CustomException

logger = get_logger(__name__)

class Prediction:
    """Handles prediction and recommendation logic"""
    
    def __init__(self, als_model, faiss_index, product_embeddings, 
                 interaction_matrix, train_matrix, mappings, 
                 warm_user_info, product_to_users_lookup, product_ids_list):
        """
        Initialize prediction component
        
        Args:
            als_model: Trained ALS model
            faiss_index: FAISS index for similarity search
            product_embeddings: Product embeddings array
            interaction_matrix: Full interaction matrix
            train_matrix: Training interaction matrix
            mappings: ID mappings dictionary
            warm_user_info: Warm user information
            product_to_users_lookup: Product-to-users lookup
            product_ids_list: List of product IDs aligned with FAISS
        """
        self.als_model = als_model
        self.faiss_index = faiss_index
        self.product_embeddings = product_embeddings
        self.interaction_matrix = interaction_matrix
        self.train_matrix = train_matrix
        self.mappings = mappings
        self.warm_user_info = warm_user_info
        self.product_to_users_lookup = product_to_users_lookup
        self.product_ids_list = product_ids_list
        
        # Create lookup mappings
        self._create_lookup_mappings()
    
    def _create_lookup_mappings(self):
        """Create efficient lookup mappings"""
        self.idx_to_user_id = self.mappings['idx_to_user_id']
        self.idx_to_product_id = self.mappings['idx_to_product_id']
        self.product_id_to_idx = self.mappings['product_id_to_idx']
        self.user_id_to_idx = self.mappings['user_id_to_idx']
        
        # Original product ID → internal product index
        self.orig_pid_to_internal_idx = {
            orig_pid: internal_idx
            for internal_idx, orig_pid in self.idx_to_product_id.items()
        }
        
        # Original product ID → FAISS index
        self.orig_pid_to_faiss_idx = {
            orig_pid: faiss_idx
            for faiss_idx, orig_pid in enumerate(self.product_ids_list)
        }
        
        # FAISS index → original product ID
        self.product_idx_to_id = {i: pid for i, pid in enumerate(self.product_ids_list)}
    
    def get_als_recommendations(self, product_idx: int, n: int = 10) -> List[Dict]:
        """
        Get ALS (collaborative filtering) recommendations
        
        Args:
            product_idx: Internal product index
            n: Number of recommendations
        
        Returns:
            List of dicts with 'user_id' and 'score'
        """
        try:
            warm_user_list = self.warm_user_info['warm_user_list']
            user_ids, scores = self.als_model.recommend(
                product_idx,
                self.train_matrix.T,
                N=min(n * 2, len(warm_user_list)),
                filter_already_liked_items=False
            )
            
            recommendations = []
            for local_idx, score in zip(user_ids, scores):
                if local_idx < len(warm_user_list):
                    global_user_idx = warm_user_list[int(local_idx)]
                    if global_user_idx in self.idx_to_user_id:
                        user_id = self.idx_to_user_id[global_user_idx]
                        recommendations.append({
                            "user_id": user_id,
                            "score": float(score)
                        })
                        if len(recommendations) >= n:
                            break
            
            return recommendations
        except Exception as e:
            logger.warning(f"Error in ALS recommendations: {e}")
            return []
    
    def get_content_based_recommendations(self, product_idx: int, n: int = 10, 
                                          top_similar: int = 50) -> List[Dict]:
        """
        Get content-based recommendations
        
        Args:
            product_idx: Internal product index
            n: Number of recommendations
            top_similar: Number of similar products to consider
        
        Returns:
            List of dicts with 'user_id' and 'score'
        """
        try:
            # Get original product ID
            orig_product_id = self.idx_to_product_id.get(product_idx)
            if orig_product_id is None:
                return []
            
            # Get FAISS index
            faiss_idx = self.orig_pid_to_faiss_idx.get(orig_product_id)
            if faiss_idx is None or faiss_idx >= len(self.product_embeddings):
                return []
            
            # Find similar products using FAISS
            query_embedding = self.product_embeddings[faiss_idx:faiss_idx+1]
            k = min(top_similar, len(self.product_ids_list))
            distances, similar_indices = self.faiss_index.search(query_embedding, k=k)
            
            # Map FAISS indices → original product IDs → internal product indices
            similar_product_indices = []
            for faiss_similar_idx in similar_indices[0]:
                if faiss_similar_idx < len(self.product_idx_to_id):
                    orig_pid = self.product_idx_to_id[faiss_similar_idx]
                    internal_idx = self.orig_pid_to_internal_idx.get(orig_pid)
                    if internal_idx is not None:
                        similar_product_indices.append(internal_idx)
            
            # Aggregate user scores from similar products
            user_scores = defaultdict(float)
            for similar_pidx in similar_product_indices:
                if similar_pidx in self.product_to_users_lookup:
                    for user_idx, score in self.product_to_users_lookup[similar_pidx]:
                        user_scores[user_idx] += float(score)
            
            # Sort and return top N
            if not user_scores:
                return []
            
            sorted_users = sorted(user_scores.items(), key=lambda x: -x[1])[:n]
            recommendations = []
            for user_idx, score in sorted_users:
                if user_idx in self.idx_to_user_id:
                    user_id = self.idx_to_user_id[user_idx]
                    recommendations.append({
                        "user_id": user_id,
                        "score": float(score)
                    })
            
            return recommendations
        except Exception as e:
            logger.warning(f"Error in content-based recommendations: {e}")
            return []
    
    def get_hybrid_recommendations(self, product_idx: int, n: int = 10,
                                   cf_weight: float = 0.4, 
                                   content_weight: float = 0.6) -> List[Dict]:
        """
        Get hybrid recommendations (ALS + content-based)
        
        Args:
            product_idx: Internal product index
            n: Number of recommendations
            cf_weight: Weight for collaborative filtering
            content_weight: Weight for content-based
        
        Returns:
            List of dicts with 'user_id' and 'score'
        """
        try:
            scores = defaultdict(float)
            
            # 1. Get CF recommendations
            try:
                product_users = self.interaction_matrix[:, product_idx].nonzero()[0]
                if len(product_users) >= 3:
                    warm_user_list = self.warm_user_info['warm_user_list']
                    user_ids, cf_scores = self.als_model.recommend(
                        product_idx,
                        self.train_matrix.T,
                        N=min(100, len(warm_user_list)),
                        filter_already_liked_items=False
                    )
                    for local_idx, score in zip(user_ids, cf_scores):
                        if local_idx < len(warm_user_list):
                            global_user_idx = warm_user_list[int(local_idx)]
                            scores[global_user_idx] += cf_weight * float(score)
            except Exception as e:
                logger.debug(f"CF recommendation failed: {e}")
            
            # 2. Get content-based recommendations
            content_recs = self.get_content_based_recommendations(product_idx, n=100)
            
            # Normalize and add content scores
            if content_recs:
                max_content_score = max([r["score"] for r in content_recs]) if content_recs else 1.0
                if max_content_score > 0:
                    for rec in content_recs:
                        user_idx = self.user_id_to_idx.get(rec["user_id"])
                        if user_idx is not None:
                            normalized_score = rec["score"] / max_content_score
                            scores[user_idx] += content_weight * normalized_score
            
            # 3. Sort and return top N
            if not scores:
                return []
            
            sorted_users = sorted(scores.items(), key=lambda x: -x[1])[:n]
            recommendations = []
            for user_idx, score in sorted_users:
                if user_idx in self.idx_to_user_id:
                    user_id = self.idx_to_user_id[user_idx]
                    recommendations.append({
                        "user_id": user_id,
                        "score": float(score)
                    })
            
            return recommendations
        except Exception as e:
            logger.warning(f"Error in hybrid recommendations: {e}")
            return []
    
    def get_popularity_recommendations(self, n: int = 10) -> List[Dict]:
        """
        Get popularity-based recommendations
        
        Args:
            n: Number of recommendations
        
        Returns:
            List of dicts with 'user_id' and 'score'
        """
        try:
            user_activity = np.array(self.interaction_matrix.sum(axis=1)).flatten()
            top_user_indices = np.argsort(user_activity)[::-1][:n]
            
            recommendations = []
            for user_idx in top_user_indices:
                if user_idx in self.idx_to_user_id:
                    user_id = self.idx_to_user_id[user_idx]
                    score = float(user_activity[user_idx])
                    recommendations.append({
                        "user_id": user_id,
                        "score": score
                    })
            
            return recommendations
        except Exception as e:
            logger.warning(f"Error in popularity recommendations: {e}")
            return []

