"""
Recommender Service - Loads artifacts and provides recommendation functions
"""
import pickle
import json
import numpy as np
import faiss
from scipy import sparse
from pathlib import Path
from collections import defaultdict
from typing import List, Dict, Optional, Tuple
import os

class RecommenderService:
    """Service class for loading artifacts and making recommendations"""
    
    def __init__(self, artifacts_dir: str = "recommender_artifacts"):
        self.artifacts_dir = Path(artifacts_dir)
        self.loaded = False
        
        # Artifacts storage
        self.als_model = None
        self.faiss_index = None
        self.product_embeddings = None
        self.tfidf_vectorizer = None
        self.svd_transformer = None
        
        # Mappings
        self.user_id_to_idx = {}
        self.idx_to_user_id = {}
        self.product_id_to_idx = {}
        self.idx_to_product_id = {}
        self.product_idx_to_id = {}
        
        # Matrices
        self.train_matrix = None
        self.interaction_matrix = None
        
        # User info
        self.warm_user_list = []
        self.warm_user_to_matrix_idx = {}
        
        # Lookups
        self.product_to_users_lookup = {}
        self.product_ids_list = []
        
        # Mappings for efficient lookups
        self.orig_pid_to_internal_idx = {}
        self.orig_pid_to_faiss_idx = {}
        
        # Config
        self.config = {}
        self.n_users = 0
        self.n_products = 0
        
        # Caching for performance
        self._recommendation_cache = {}
        self._similar_products_cache = {}
        self._cache_max_size = 1000  # Limit cache size
        
        # Valid IDs sets for fast lookup
        self._valid_product_ids = set()
        self._valid_user_ids = set()
    
    def load_artifacts(self):
        """Load all saved artifacts"""
        if not self.artifacts_dir.exists():
            raise FileNotFoundError(
                f"Artifacts directory '{self.artifacts_dir}' not found. "
                "Please run the training pipeline first: python run_pipeline.py"
            )
        
        als_model_path = self.artifacts_dir / 'als_model.pkl'
        if not als_model_path.exists():
            raise FileNotFoundError(
                f"Artifacts not found in '{self.artifacts_dir}'. "
                "Please run the training pipeline first: python run_pipeline.py"
            )
        
        print(f"📁 Loading artifacts from: {self.artifacts_dir}/")
        
        # 1. Load ALS model
        print("1️⃣ Loading ALS Model...")
        with open(als_model_path, 'rb') as f:
            self.als_model = pickle.load(f)
        
        # 2. Load FAISS index
        print("2️⃣ Loading FAISS Index...")
        self.faiss_index = faiss.read_index(str(self.artifacts_dir / 'faiss_index.bin'))
        
        # 3. Load embeddings
        print("3️⃣ Loading Embeddings...")
        self.product_embeddings = np.load(self.artifacts_dir / 'product_embeddings.npy')
        
        # 4. Load TF-IDF and SVD
        print("4️⃣ Loading TF-IDF and SVD...")
        with open(self.artifacts_dir / 'tfidf_vectorizer.pkl', 'rb') as f:
            self.tfidf_vectorizer = pickle.load(f)
        with open(self.artifacts_dir / 'svd_transformer.pkl', 'rb') as f:
            self.svd_transformer = pickle.load(f)
        
        # 5. Load ID mappings
        print("5️⃣ Loading ID Mappings...")
        with open(self.artifacts_dir / 'id_mappings.json', 'r') as f:
            mappings = json.load(f)
        
        self.user_id_to_idx = {int(k): int(v) for k, v in mappings['user_id_to_idx'].items()}
        self.idx_to_user_id = {int(k): int(v) for k, v in mappings['idx_to_user_id'].items()}
        self.product_id_to_idx = {int(k): int(v) for k, v in mappings['product_id_to_idx'].items()}
        self.idx_to_product_id = {int(k): int(v) for k, v in mappings['idx_to_product_id'].items()}
        self.n_users = mappings['n_users']
        self.n_products = mappings['n_products']
        
        # Load product_idx_to_id if available, otherwise create from product_ids_list
        if 'product_idx_to_id' in mappings:
            self.product_idx_to_id = {int(k): int(v) for k, v in mappings['product_idx_to_id'].items()}
        else:
            # Create from product_ids_list (FAISS index → original product ID)
            self.product_idx_to_id = {i: pid for i, pid in enumerate(self.product_ids_list)}
        
        # 6. Load sparse matrices
        print("6️⃣ Loading Sparse Matrices...")
        self.train_matrix = sparse.load_npz(str(self.artifacts_dir / 'train_matrix.npz'))
        self.interaction_matrix = sparse.load_npz(str(self.artifacts_dir / 'interaction_matrix.npz'))
        
        # 7. Load warm user info
        print("7️⃣ Loading Warm User Info...")
        with open(self.artifacts_dir / 'warm_user_info.json', 'r') as f:
            warm_user_info = json.load(f)
        self.warm_user_list = [int(u) for u in warm_user_info['warm_user_list']]
        self.warm_user_to_matrix_idx = {int(k): int(v) for k, v in warm_user_info['warm_user_to_matrix_idx'].items()}
        
        # 8. Load product-to-users lookup
        print("8️⃣ Loading Product-User Lookup...")
        with open(self.artifacts_dir / 'product_to_users_lookup.json', 'r') as f:
            product_to_users_data = json.load(f)
        self.product_to_users_lookup = {
            int(k): [(int(u), float(s)) for u, s in v]
            for k, v in product_to_users_data.items()
        }
        
        # 9. Load product IDs list
        print("9️⃣ Loading Product IDs List...")
        with open(self.artifacts_dir / 'product_ids_list.json', 'r') as f:
            self.product_ids_list = [int(pid) for pid in json.load(f)]
        
        # 10. Load configuration
        print("🔟 Loading Configuration...")
        with open(self.artifacts_dir / 'config.json', 'r') as f:
            self.config = json.load(f)
        
        # Create product_idx_to_id if not loaded (FAISS index → original product ID)
        if not hasattr(self, 'product_idx_to_id') or not self.product_idx_to_id:
            self.product_idx_to_id = {i: pid for i, pid in enumerate(self.product_ids_list)}
        
        # Create efficient lookup mappings
        self._create_lookup_mappings()
        
        # Build valid IDs sets for fast validation
        self._valid_product_ids = set(self.product_id_to_idx.keys())
        self._valid_user_ids = set(self.user_id_to_idx.keys())
        
        print(f"✅ Valid IDs: {len(self._valid_product_ids):,} products, {len(self._valid_user_ids):,} users")
        
        self.loaded = True
        print("✅ All artifacts loaded successfully!")
    
    def _create_lookup_mappings(self):
        """Create efficient lookup mappings"""
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
    
    def get_recommendations(
        self,
        product_id: int,
        n: int = 10,
        method: str = "hybrid"
    ) -> List[Dict]:
        """
        Get top-N user recommendations for a product.
        
        Args:
            product_id: Product ID to get recommendations for
            n: Number of recommendations
            method: 'als', 'content', 'hybrid', or 'popularity'
        
        Returns:
            List of dicts with 'user_id' and 'score'
        """
        if not self.loaded:
            raise RuntimeError("Artifacts not loaded. Call load_artifacts() first.")
        
        # Fast validation using set lookup
        if product_id not in self._valid_product_ids:
            return []
        
        # Check cache
        cache_key = (product_id, n, method)
        if cache_key in self._recommendation_cache:
            return self._recommendation_cache[cache_key]
        
        # Validate n
        n = max(1, min(n, 100))  # Clamp between 1 and 100
        
        product_idx = self.product_id_to_idx[product_id]
        
        try:
            if method == "als":
                recommendations = self._get_als_recommendations(product_idx, n)
            elif method == "content":
                recommendations = self._get_content_based_recommendations(product_idx, n)
            elif method == "hybrid":
                recommendations = self._get_hybrid_recommendations(product_idx, n)
            elif method == "popularity":
                recommendations = self._get_popularity_recommendations(n)
            else:
                raise ValueError(f"Unknown method: {method}. Use 'als', 'content', 'hybrid', or 'popularity'")
            
            # Cache results (limit cache size)
            if len(self._recommendation_cache) < self._cache_max_size:
                self._recommendation_cache[cache_key] = recommendations
            
            return recommendations
        except Exception as e:
            # Log error but return empty list instead of raising
            print(f"Error in get_recommendations for product {product_id}, method {method}: {e}")
            return []
    
    def _get_als_recommendations(self, product_idx: int, n: int) -> List[Dict]:
        """Get ALS (collaborative filtering) recommendations - optimized with error handling"""
        try:
            # Limit N to prevent memory issues
            max_n = min(n * 2, len(self.warm_user_list), 100)
            if max_n <= 0:
                return []
            
            user_ids, scores = self.als_model.recommend(
                product_idx,
                self.train_matrix.T,
                N=max_n,
                filter_already_liked_items=False
            )
            
            recommendations = []
            for local_idx, score in zip(user_ids, scores):
                try:
                    local_idx_int = int(local_idx)
                    if local_idx_int < len(self.warm_user_list):
                        global_user_idx = self.warm_user_list[local_idx_int]
                        if global_user_idx in self.idx_to_user_id:
                            user_id = self.idx_to_user_id[global_user_idx]
                            recommendations.append({
                                "user_id": user_id,
                                "score": float(score)
                            })
                            if len(recommendations) >= n:
                                break
                except (IndexError, KeyError, ValueError):
                    continue  # Skip invalid indices
            
            return recommendations
        except Exception as e:
            # Silent fail - return empty list instead of raising
            return []
    
    def _get_content_based_recommendations(self, product_idx: int, n: int) -> List[Dict]:
        """Get content-based recommendations"""
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
        k = min(50, len(self.product_ids_list))
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
    
    def _get_hybrid_recommendations(self, product_idx: int, n: int) -> List[Dict]:
        """Get hybrid recommendations (ALS + content-based) - optimized with error handling"""
        scores = defaultdict(float)
        cf_weight = 0.4
        content_weight = 0.6
        
        # 1. Get CF recommendations (with error handling)
        try:
            product_users = self.interaction_matrix[:, product_idx].nonzero()[0]
            if len(product_users) >= 3:
                try:
                    user_ids, cf_scores = self.als_model.recommend(
                        product_idx,
                        self.train_matrix.T,
                        N=min(100, len(self.warm_user_list)),
                        filter_already_liked_items=False
                    )
                    for local_idx, score in zip(user_ids, cf_scores):
                        if local_idx < len(self.warm_user_list):
                            global_user_idx = self.warm_user_list[int(local_idx)]
                            scores[global_user_idx] += cf_weight * float(score)
                except Exception as e:
                    # If ALS fails, continue with content-based only
                    print(f"ALS recommendation failed for product_idx {product_idx}: {e}")
        except Exception as e:
            print(f"Error checking product users for product_idx {product_idx}: {e}")
        
        # 2. Get content-based recommendations
        try:
            content_users = self._get_content_based_recommendations(product_idx, n=100)
            
            # Normalize and add content scores
            if content_users:
                max_content_score = max([r["score"] for r in content_users]) if content_users else 1.0
                if max_content_score > 0:
                    for rec in content_users:
                        user_idx = self.user_id_to_idx.get(rec["user_id"])
                        if user_idx is not None:
                            normalized_score = rec["score"] / max_content_score
                            scores[user_idx] += content_weight * normalized_score
        except Exception as e:
            print(f"Content-based recommendation failed for product_idx {product_idx}: {e}")
        
        # 3. Sort and return top N
        if not scores:
            return []
        
        try:
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
            print(f"Error sorting hybrid recommendations: {e}")
            return []
    
    def _get_popularity_recommendations(self, n: int) -> List[Dict]:
        """Get popularity-based recommendations"""
        # Use most active users from interaction matrix
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
    
    def get_similar_products(self, product_id: int, n: int = 10) -> List[Dict]:
        """
        Get top-N similar products based on embeddings.
        
        Args:
            product_id: Product ID to find similar products for
            n: Number of similar products to return
        
        Returns:
            List of dicts with 'product_id' and 'similarity_score'
        """
        if not self.loaded:
            raise RuntimeError("Artifacts not loaded. Call load_artifacts() first.")
        
        # Fast validation
        if product_id not in self._valid_product_ids:
            return []
        
        # Check cache
        cache_key = (product_id, n)
        if cache_key in self._similar_products_cache:
            return self._similar_products_cache[cache_key]
        
        # Check if product has FAISS index
        if product_id not in self.orig_pid_to_faiss_idx:
            return []
        
        faiss_idx = self.orig_pid_to_faiss_idx[product_id]
        
        if faiss_idx >= len(self.product_embeddings):
            return []
        
        # Validate n
        n = max(1, min(n, 50))  # Clamp between 1 and 50
        
        # Find similar products using FAISS
        query_embedding = self.product_embeddings[faiss_idx:faiss_idx+1]
        k = min(n + 1, len(self.product_ids_list))  # +1 to exclude self
        distances, similar_indices = self.faiss_index.search(query_embedding, k=k)
        
        similar_products = []
        for i, (idx, dist) in enumerate(zip(similar_indices[0], distances[0])):
            if idx < len(self.product_idx_to_id):
                similar_pid = self.product_idx_to_id[idx]
                # Skip the product itself
                if similar_pid == product_id:
                    continue
                
                # Convert distance to similarity (lower distance = higher similarity)
                similarity = 1.0 / (1.0 + float(dist))
                similar_products.append({
                    "product_id": similar_pid,
                    "similarity_score": similarity,
                    "distance": float(dist)
                })
                
                if len(similar_products) >= n:
                    break
        
        # Cache results
        if len(self._similar_products_cache) < self._cache_max_size:
            self._similar_products_cache[cache_key] = similar_products
        
        return similar_products
    
    def get_user_history(self, user_id: int) -> Dict:
        """
        Get interaction history for a user.
        
        Args:
            user_id: User ID to get history for
        
        Returns:
            Dict with history list and statistics, or None if user not found
        """
        if not self.loaded:
            raise RuntimeError("Artifacts not loaded. Call load_artifacts() first.")
        
        # Fast validation
        if user_id not in self._valid_user_ids:
            return None
        
        user_idx = self.user_id_to_idx[user_id]
        
        # Get user's interactions from sparse matrix
        user_interactions = self.interaction_matrix[user_idx, :]
        
        # Get non-zero products (products user interacted with)
        product_indices = user_interactions.nonzero()[1]
        scores = user_interactions.data
        
        if len(product_indices) == 0:
            return {
                "user_id": user_id,
                "history": [],
                "stats": {
                    "total_interactions": 0,
                    "unique_products": 0,
                    "event_types": 0
                }
            }
        
        # Build history list
        history = []
        for product_idx, score in zip(product_indices, scores):
            if product_idx in self.idx_to_product_id:
                product_id = self.idx_to_product_id[product_idx]
                history.append({
                    "product_id": product_id,
                    "score": float(score),
                    "event": "interaction",  # Generic event type
                    "event_date": None  # Date not available in sparse matrix
                })
        
        # Sort by score (descending)
        history.sort(key=lambda x: x["score"], reverse=True)
        
        # Calculate statistics
        unique_products = len(set(h["product_id"] for h in history))
        
        return {
            "user_id": user_id,
            "history": history,
            "stats": {
                "total_interactions": len(history),
                "unique_products": unique_products,
                "event_types": 1  # We only have generic "interaction" from sparse matrix
            }
        }
    
    def get_stats(self) -> Dict:
        """Get system statistics"""
        return {
            "total_users": self.n_users,
            "total_products": self.n_products,
            "warm_users": len(self.warm_user_list),
            "cold_users": self.n_users - len(self.warm_user_list),
            "embedding_dimension": int(self.product_embeddings.shape[1]) if self.product_embeddings is not None else 0,
            "faiss_index_size": int(self.faiss_index.ntotal) if self.faiss_index is not None else 0,
            "loaded": self.loaded
        }

