"""
Unit tests for model trainer component
Run this file directly: python tests/test_model_trainer.py
"""
import os
import sys
import numpy as np
import pandas as pd


# Add parent directory to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.components.model_trainer import ModelTrainer

def test_model_trainer_initialization():
    """Test ModelTrainer initialization"""
    config = {
        "model": {
            "random_seed": 42,
            "als": {"factors": 64, "regularization": 0.3, "iterations": 30, "alpha": 40},
            "embeddings": {"max_features": 100, "ngram_range": [1, 2], "min_df": 3, "n_components": 50}
        }
    }
    mt = ModelTrainer(config)
    assert mt.config == config
    assert mt.random_seed == 42

def test_create_product_embeddings():
    """Test product embedding creation"""
    config = {
        "model": {
            "random_seed": 42,
            "embeddings": {"max_features": 10, "ngram_range": [1, 1], "min_df": 1, "n_components": 5}
        }
    }
    mt = ModelTrainer(config)
    
    # Use more diverse text to ensure we have at least 5 unique features
    product_info = pd.DataFrame({
        'product_id': [1, 2, 3, 4],
        'cleaned_text': ['laptop computer tech', 'smartphone mobile device', 'headphones audio sound', 'keyboard mouse input']
    })
    
    embeddings, tfidf, svd, product_info_final = mt.create_product_embeddings(product_info)
    
    # Check embeddings shape
    assert embeddings.shape[0] == 4  # Number of products
    
    # Number of components should be min(n_components, n_features, n_documents)
    # SVD can produce at most min(n_features, n_documents) components
    # Verify it doesn't exceed requested n_components, available features, or documents
    n_features = len(tfidf.vocabulary_)
    n_documents = embeddings.shape[0]
    expected_components = min(mt.n_components, n_features, n_documents)
    assert embeddings.shape[1] == expected_components
    assert embeddings.shape[1] <= mt.n_components  # Should not exceed requested n_components
    assert embeddings.shape[1] <= n_features  # Should not exceed available features
    assert embeddings.shape[1] <= n_documents  # Should not exceed number of documents

def test_build_faiss_index():
    """Test FAISS index building"""
    config = {"model": {"random_seed": 42, "embeddings": {}}}
    mt = ModelTrainer(config)
    
    embeddings = np.random.rand(10, 50).astype('float32')
    index = mt.build_faiss_index(embeddings)
    assert index.ntotal == 10
    assert index.d == 50

if __name__ == "__main__":
    test_model_trainer_initialization()
    print("✅ Model trainer initialization test passed!")
    
    test_create_product_embeddings()
    print("✅ Product embeddings test passed!")
    
    test_build_faiss_index()
    print("✅ FAISS index test passed!")
    
    print("\n✅ All model trainer tests passed!")
