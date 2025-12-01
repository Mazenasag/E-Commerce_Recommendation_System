"""
Training Pipeline
End-to-end pipeline for training the recommender system
"""
import pandas as pd
import numpy as np
import pickle
import json
import faiss
from pathlib import Path
from scipy import sparse
from datetime import datetime
from src.components.data_ingestion import DataIngestion
from src.components.data_transformation import DataTransformation
from src.components.model_trainer import ModelTrainer
from src.components.model_evaluator import ModelEvaluator
from src.config_loader import load_config, save_config
import sys
from src.utils.logger import get_logger
from src.utils.exception import CustomException
from src.utils.helpers import ensure_dir

logger = get_logger(__name__)

def run_training_pipeline(config_path: str = "config/config.yaml"):
    """
    Run the complete training pipeline
    
    Args:
        config_path: Path to configuration file
    """
    logger.info("=" * 70)
    logger.info("🚀 STARTING TRAINING PIPELINE")
    logger.info("=" * 70)
    
    # Load configuration
    config = load_config(config_path)
    artifacts_config = config.get('artifacts', {})
    artifacts_base = Path(artifacts_config.get('base_path', 'artifacts'))
    ensure_dir(artifacts_base)
    
    # Step 1: Data Ingestion
    logger.info("\n" + "=" * 70)
    logger.info("STEP 1: DATA INGESTION")
    logger.info("=" * 70)
    data_ingestion = DataIngestion(config)
    df = data_ingestion.load_data()
    data_ingestion.validate_data(df)
    
    # Step 2: Data Transformation
    logger.info("\n" + "=" * 70)
    logger.info("STEP 2: DATA TRANSFORMATION")
    logger.info("=" * 70)
    data_transformation = DataTransformation(config)
    
    # Clean data
    df = data_transformation.clean_data(df)
    
    # Process product names
    df = data_transformation.process_product_names(df)
    
    # Create weighted scores
    df = data_transformation.create_weighted_scores(df)
    
    # Create ID mappings
    mappings = data_transformation.create_id_mappings(df)
    
    # Identify user segments
    warm_users, cold_users = data_transformation.identify_user_segments(df)
    
    # Save processed data
    processed_path = config.get('data', {}).get('processed_data_path', 'data/processed/processed_data.parquet')
    data_transformation.save_processed_data(df, processed_path)
    
    # Step 3: Train/Test Split
    logger.info("\n" + "=" * 70)
    logger.info("STEP 3: TRAIN/TEST SPLIT")
    logger.info("=" * 70)
    train_ratio = config.get('data', {}).get('train_test_split_ratio', 0.8)
    date_min = df['event_date'].min()
    date_max = df['event_date'].max()
    date_range = (date_max - date_min).days
    split_date = date_min + pd.Timedelta(days=int(date_range * train_ratio))
    
    train_df = df[df['event_date'] < split_date].copy()
    test_df = df[df['event_date'] >= split_date].copy()
    
    logger.info(f"   Split date: {split_date.date()}")
    logger.info(f"   Train: {len(train_df):,} ({len(train_df)/len(df)*100:.1f}%)")
    logger.info(f"   Test: {len(test_df):,} ({len(test_df)/len(df)*100:.1f}%)")
    
    # Step 4: Model Training
    logger.info("\n" + "=" * 70)
    logger.info("STEP 4: MODEL TRAINING")
    logger.info("=" * 70)
    model_trainer = ModelTrainer(config)
    
    # 4.1: Create product embeddings
    product_info = df[['product_id', 'cleaned_text']].drop_duplicates(subset=['product_id'])
    product_embeddings, tfidf_vectorizer, svd_transformer = model_trainer.create_product_embeddings(product_info)
    
    # 4.2: Build FAISS index
    faiss_index = model_trainer.build_faiss_index(product_embeddings)
    
    # 4.3: Build interaction matrices
    # Full matrix (for content-based)
    interaction_matrix, _ = model_trainer.build_interaction_matrix(train_df, mappings)
    
    # Warm user matrix (for ALS)
    warm_user_indices = set([mappings['user_id_to_idx'][uid] for uid in warm_users if uid in mappings['user_id_to_idx']])
    train_warm = train_df[train_df['user_idx'].isin(warm_user_indices)].copy()
    train_matrix, warm_user_info = model_trainer.build_interaction_matrix(
        train_warm, mappings, user_segment=warm_users
    )
    
    # 4.4: Train ALS model
    als_model = model_trainer.train_als_model(train_matrix)
    
    # 4.5: Build product-user lookup
    product_to_users_lookup = model_trainer.build_product_user_lookup(interaction_matrix, mappings)
    
    # Step 5: Save Artifacts
    logger.info("\n" + "=" * 70)
    logger.info("STEP 5: SAVING ARTIFACTS")
    logger.info("=" * 70)
    
    # Create product IDs list aligned with FAISS
    product_ids_list = product_info['product_id'].tolist()
    
    # Save all artifacts
    save_artifacts(
        als_model=als_model,
        faiss_index=faiss_index,
        product_embeddings=product_embeddings,
        tfidf_vectorizer=tfidf_vectorizer,
        svd_transformer=svd_transformer,
        mappings=mappings,
        train_matrix=train_matrix,
        interaction_matrix=interaction_matrix,
        warm_user_info=warm_user_info,
        product_to_users_lookup=product_to_users_lookup,
        product_ids_list=product_ids_list,
        config=config,
        artifacts_base=artifacts_base
    )
    
    logger.info("\n" + "=" * 70)
    logger.info("✅ TRAINING PIPELINE COMPLETE!")
    logger.info("=" * 70)
    
    return {
        'als_model': als_model,
        'faiss_index': faiss_index,
        'product_embeddings': product_embeddings,
        'mappings': mappings,
        'train_matrix': train_matrix,
        'interaction_matrix': interaction_matrix,
        'warm_user_info': warm_user_info,
        'product_to_users_lookup': product_to_users_lookup,
        'product_ids_list': product_ids_list
    }

def save_artifacts(als_model, faiss_index, product_embeddings, tfidf_vectorizer,
                   svd_transformer, mappings, train_matrix, interaction_matrix,
                   warm_user_info, product_to_users_lookup, product_ids_list,
                   config, artifacts_base: Path):
    """Save all training artifacts"""
    
    # 1. Save ALS model
    logger.info("1️⃣ Saving ALS model...")
    with open(artifacts_base / 'als_model.pkl', 'wb') as f:
        pickle.dump(als_model, f)
    
    # 2. Save FAISS index
    logger.info("2️⃣ Saving FAISS index...")
    faiss.write_index(faiss_index, str(artifacts_base / 'faiss_index.bin'))
    
    # 3. Save embeddings
    logger.info("3️⃣ Saving embeddings...")
    np.save(artifacts_base / 'product_embeddings.npy', product_embeddings)
    
    # 4. Save TF-IDF and SVD
    logger.info("4️⃣ Saving TF-IDF and SVD...")
    with open(artifacts_base / 'tfidf_vectorizer.pkl', 'wb') as f:
        pickle.dump(tfidf_vectorizer, f)
    with open(artifacts_base / 'svd_transformer.pkl', 'wb') as f:
        pickle.dump(svd_transformer, f)
    
    # 5. Save ID mappings
    logger.info("5️⃣ Saving ID mappings...")
    mappings_serializable = {
        'user_id_to_idx': {str(k): int(v) for k, v in mappings['user_id_to_idx'].items()},
        'idx_to_user_id': {str(k): int(v) for k, v in mappings['idx_to_user_id'].items()},
        'product_id_to_idx': {str(k): int(v) for k, v in mappings['product_id_to_idx'].items()},
        'idx_to_product_id': {str(k): int(v) for k, v in mappings['idx_to_product_id'].items()},
        'n_users': int(mappings['n_users']),
        'n_products': int(mappings['n_products'])
    }
    with open(artifacts_base / 'id_mappings.json', 'w') as f:
        json.dump(mappings_serializable, f, indent=2)
    
    # 6. Save sparse matrices
    logger.info("6️⃣ Saving sparse matrices...")
    sparse.save_npz(artifacts_base / 'train_matrix.npz', train_matrix)
    sparse.save_npz(artifacts_base / 'interaction_matrix.npz', interaction_matrix)
    
    # 7. Save warm user info
    logger.info("7️⃣ Saving warm user info...")
    warm_user_info_serializable = {
        'warm_user_list': [int(u) for u in warm_user_info['warm_user_list']],
        'warm_user_to_matrix_idx': {str(k): int(v) for k, v in warm_user_info['warm_user_to_matrix_idx'].items()}
    }
    with open(artifacts_base / 'warm_user_info.json', 'w') as f:
        json.dump(warm_user_info_serializable, f, indent=2)
    
    # 8. Save product-to-users lookup
    logger.info("8️⃣ Saving product-user lookup...")
    product_to_users_serializable = {
        str(k): [(int(u), float(s)) for u, s in v]
        for k, v in product_to_users_lookup.items()
    }
    with open(artifacts_base / 'product_to_users_lookup.json', 'w') as f:
        json.dump(product_to_users_serializable, f)
    
    # 9. Save product IDs list
    logger.info("9️⃣ Saving product IDs list...")
    with open(artifacts_base / 'product_ids_list.json', 'w') as f:
        json.dump([str(pid) for pid in product_ids_list], f)
    
    # 10. Save configuration
    logger.info("🔟 Saving configuration...")
    config_to_save = {
        'event_weights': config.get('event_weights', {}),
        'embedding_dimension': int(product_embeddings.shape[1]),
        'n_components_svd': int(svd_transformer.n_components),
        'explained_variance': float(svd_transformer.explained_variance_ratio_.sum()),
        'faiss_index_type': 'IndexFlatL2',
        'model_hyperparameters': {
            'als_factors': config.get('model', {}).get('als', {}).get('factors', 64),
            'als_regularization': config.get('model', {}).get('als', {}).get('regularization', 0.3),
            'als_iterations': config.get('model', {}).get('als', {}).get('iterations', 30)
        },
        'data_stats': {
            'total_users': int(mappings['n_users']),
            'total_products': int(mappings['n_products']),
            'warm_users': int(len(warm_user_info['warm_user_list'])),
            'cold_users': int(mappings['n_users'] - len(warm_user_info['warm_user_list']))
        },
        'training_date': datetime.now().isoformat()
    }
    with open(artifacts_base / 'config.json', 'w') as f:
        json.dump(config_to_save, f, indent=2)
    
    logger.info("✅ All artifacts saved!")

if __name__ == "__main__":
    run_training_pipeline()

