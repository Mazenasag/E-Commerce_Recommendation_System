
# Recommender System Artifacts Summary

## Files Saved:
1. als_model.pkl - Trained ALS model for collaborative filtering
2. faiss_index.bin - FAISS index for fast product similarity search
3. product_embeddings.npy - Product name embeddings (TF-IDF + SVD)
4. tfidf_vectorizer.pkl - TF-IDF vectorizer
5. svd_transformer.pkl - SVD dimension reducer
6. id_mappings.json - User and product ID mappings
7. train_matrix.npz - Training interaction matrix (sparse)
8. interaction_matrix.npz - Full interaction matrix (sparse)
9. warm_user_info.json - Warm user indices and mappings
10. product_to_users_lookup.json - Product-to-users lookup table
11. product_ids_list.json - Product IDs aligned with FAISS index
12. config.json - System configuration and hyperparameters

## System Info:
- Total Users: 433,787
- Total Products: 200,325
- Warm Users (2+ interactions): 49,359
- Cold Users (1 interaction): 384,428
- Embedding Dimension: 50
- FAISS Index Size: 200,325 vectors

## Usage:
See the recommender class in Cell 12 for how to load and use these artifacts.
