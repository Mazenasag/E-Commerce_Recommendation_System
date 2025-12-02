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
    
    # Check if processed data exists
    processed_path = config.get('data', {}).get('processed_data_path', 'data/processed/processed_data.csv')
    csv_path = processed_path.replace('.parquet', '.csv') if processed_path.endswith('.parquet') else processed_path + '.csv' if not processed_path.endswith('.csv') else processed_path
    
    if Path(csv_path).exists():
        logger.info(f"📂 Loading preprocessed data from {csv_path}...")
        df = data_transformation.load_processed_data(processed_path)
        
        # Recreate ID mappings (needed for model training)
        logger.info("🔄 Recreating ID mappings...")
        mappings = data_transformation.create_id_mappings(df)
        
        # Reidentify user segments
        logger.info("🔄 Reidentifying user segments...")
        warm_users, cold_users = data_transformation.identify_user_segments(df)
        
        logger.info("✅ Using preprocessed data, skipping transformation steps")
    else:
        logger.info("🔄 Preprocessed data not found, running transformation...")
        
        # Clean data
        df = data_transformation.clean_data(df)
        
        # Process product names
        df = data_transformation.process_product_names(df)
        
        # Create weighted scores
        df = data_transformation.create_weighted_scores(df)
        
        # Create ID mappings (adds user_idx and product_idx to df)
        mappings = data_transformation.create_id_mappings(df)
        
        # Identify user segments
        warm_users, cold_users = data_transformation.identify_user_segments(df)
        
        # Save processed data as CSV
        logger.info(f"💾 Saving preprocessed data to {csv_path}...")
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
    product_embeddings, tfidf_vectorizer, svd_transformer, product_info_final = model_trainer.create_product_embeddings(product_info)
    
    # Create product_id_to_embedding mapping (EXACTLY like notebook line 1231)
    # Notebook: product_id_to_embedding = dict(zip(product_info['product_id'], product_embeddings_reduced))
    product_id_to_embedding = dict(zip(product_info_final['product_id'], product_embeddings))
    
    # Create product_ids_list (EXACTLY like notebook line 1300)
    # Notebook: product_ids_list = list(product_id_to_embedding.keys())
    product_ids_list = list(product_id_to_embedding.keys())
    
    # Create product_idx_to_id and product_id_to_idx (EXACTLY like notebook lines 1301-1302)
    # Notebook: product_idx_to_id = {i: pid for i, pid in enumerate(product_ids_list)}
    # Notebook: product_id_to_idx = {pid: i for i, pid in enumerate(product_ids_list)}
    product_idx_to_id = {i: pid for i, pid in enumerate(product_ids_list)}
    product_id_to_idx_embeddings = {pid: i for i, pid in enumerate(product_ids_list)}
    
    # Create all_embeddings (EXACTLY like notebook line 1306)
    # Notebook: all_embeddings = np.vstack([product_id_to_embedding[pid] for pid in product_ids_list])
    all_embeddings = np.vstack([product_id_to_embedding[pid] for pid in product_ids_list])
    
    # Create products_with_embeddings (EXACTLY like notebook line 2767)
    # Notebook: products_with_embeddings = set(product_id_to_idx.keys())
    products_with_embeddings = set(product_id_to_idx_embeddings.keys())
    
    logger.info(f"   📊 Products with embeddings: {len(products_with_embeddings):,}")
    logger.info(f"   📊 All embeddings shape: {all_embeddings.shape}")
    
    # 4.2: Build FAISS index (using all_embeddings, EXACTLY like notebook)
    faiss_index = model_trainer.build_faiss_index(all_embeddings)
    
    # 4.3: Build interaction matrices
    interaction_matrix, _ = model_trainer.build_interaction_matrix(train_df, mappings)
    
    warm_user_indices = set([mappings['user_id_to_idx'][uid] for uid in warm_users if uid in mappings['user_id_to_idx']])
    train_warm = train_df[train_df['user_idx'].isin(warm_user_indices)].copy()
    train_matrix, warm_user_info = model_trainer.build_interaction_matrix(
        train_warm, mappings, user_segment=warm_users
    )
    
    # 4.4: Train ALS model
    als_model = model_trainer.train_als_model(train_matrix)
    
    # 4.5: Build product-user lookup (EXACTLY like notebook)
    # Notebook builds from interaction_matrix (COO format) and filters by products_with_embeddings
    product_to_users_lookup = model_trainer.build_product_user_lookup(interaction_matrix, mappings, products_with_embeddings)
    
    # Step 5: Save Artifacts
    logger.info("\n" + "=" * 70)
    logger.info("STEP 5: SAVING ARTIFACTS")
    logger.info("=" * 70)
    
    # Save all artifacts (EXACTLY like notebook Cell 13)
    save_artifacts(
        als_model=als_model,
        faiss_index=faiss_index,
        all_embeddings=all_embeddings,
        tfidf_vectorizer=tfidf_vectorizer,
        svd_transformer=svd_transformer,
        mappings=mappings,
        train_matrix=train_matrix,
        interaction_matrix=interaction_matrix,
        warm_user_info=warm_user_info,
        product_to_users_lookup=product_to_users_lookup,
        product_ids_list=product_ids_list,
        product_idx_to_id=product_idx_to_id,
        df=df,
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

def check_artifact_exists(artifacts_base: Path, filename: str) -> bool:
    """Check if an artifact file exists"""
    return (artifacts_base / filename).exists()

def check_all_artifacts_exist(artifacts_base: Path) -> bool:
    """Check if all required artifacts exist"""
    required_artifacts = [
        'als_model.pkl',
        'faiss_index.bin',
        'product_embeddings.npy',
        'tfidf_vectorizer.pkl',
        'svd_transformer.pkl',
        'id_mappings.json',
        'train_matrix.npz',
        'interaction_matrix.npz',
        'warm_user_info.json',
        'product_to_users_lookup.json',
        'product_ids_list.json',
        'config.json'
    ]
    return all(check_artifact_exists(artifacts_base, artifact) for artifact in required_artifacts)

def save_artifacts(als_model, faiss_index, all_embeddings, tfidf_vectorizer,
                   svd_transformer, mappings, train_matrix, interaction_matrix,
                   warm_user_info, product_to_users_lookup, product_ids_list,
                   product_idx_to_id, df, config, artifacts_base: Path, force_recreate: bool = False):
    """Save all training artifacts, skipping if they already exist"""
    
    saved_count = 0
    skipped_count = 0
    
    # 1. Save ALS model
    als_path = artifacts_base / 'als_model.pkl'
    if not check_artifact_exists(artifacts_base, 'als_model.pkl') or force_recreate:
        logger.info("1️⃣ Saving ALS model...")
        with open(als_path, 'wb') as f:
            pickle.dump(als_model, f)
        saved_count += 1
    else:
        logger.info("1️⃣ ALS model already exists, skipping...")
        skipped_count += 1
    
    # 2. Save FAISS index
    faiss_path = artifacts_base / 'faiss_index.bin'
    if not check_artifact_exists(artifacts_base, 'faiss_index.bin') or force_recreate:
        logger.info("2️⃣ Saving FAISS index...")
        faiss.write_index(faiss_index, str(faiss_path))
        saved_count += 1
    else:
        logger.info("2️⃣ FAISS index already exists, skipping...")
        skipped_count += 1
    
    # 3. Save embeddings (EXACTLY like notebook: save all_embeddings)
    embeddings_path = artifacts_base / 'product_embeddings.npy'
    if not check_artifact_exists(artifacts_base, 'product_embeddings.npy') or force_recreate:
        logger.info("3️⃣ Saving embeddings...")
        np.save(embeddings_path, all_embeddings)
        saved_count += 1
    else:
        logger.info("3️⃣ Embeddings already exist, skipping...")
        skipped_count += 1
    
    # 4. Save TF-IDF and SVD
    tfidf_path = artifacts_base / 'tfidf_vectorizer.pkl'
    svd_path = artifacts_base / 'svd_transformer.pkl'
    if not check_artifact_exists(artifacts_base, 'tfidf_vectorizer.pkl') or force_recreate:
        logger.info("4️⃣ Saving TF-IDF vectorizer...")
        with open(tfidf_path, 'wb') as f:
            pickle.dump(tfidf_vectorizer, f)
        saved_count += 1
    else:
        logger.info("4️⃣ TF-IDF vectorizer already exists, skipping...")
        skipped_count += 1
    
    if not check_artifact_exists(artifacts_base, 'svd_transformer.pkl') or force_recreate:
        logger.info("   Saving SVD transformer...")
        with open(svd_path, 'wb') as f:
            pickle.dump(svd_transformer, f)
        saved_count += 1
    else:
        logger.info("   SVD transformer already exists, skipping...")
        skipped_count += 1
    
    # 5. Save ID mappings (EXACTLY like notebook: include product_idx_to_id)
    mappings_path = artifacts_base / 'id_mappings.json'
    if not check_artifact_exists(artifacts_base, 'id_mappings.json') or force_recreate:
        logger.info("5️⃣ Saving ID mappings...")
        mappings_serializable = {
            'user_id_to_idx': {str(k): int(v) for k, v in mappings['user_id_to_idx'].items()},
            'idx_to_user_id': {str(k): int(v) for k, v in mappings['idx_to_user_id'].items()},
            'product_id_to_idx': {str(k): int(v) for k, v in mappings['product_id_to_idx'].items()},
            'idx_to_product_id': {str(k): int(v) for k, v in mappings['idx_to_product_id'].items()},
            'product_idx_to_id': {str(k): int(v) for k, v in product_idx_to_id.items()},
            'n_users': int(mappings['n_users']),
            'n_products': int(mappings['n_products'])
        }
        with open(mappings_path, 'w') as f:
            json.dump(mappings_serializable, f, indent=2)
        saved_count += 1
    else:
        logger.info("5️⃣ ID mappings already exist, skipping...")
        skipped_count += 1
    
    # 6. Save sparse matrices
    train_matrix_path = artifacts_base / 'train_matrix.npz'
    interaction_matrix_path = artifacts_base / 'interaction_matrix.npz'
    if not check_artifact_exists(artifacts_base, 'train_matrix.npz') or force_recreate:
        logger.info("6️⃣ Saving train matrix...")
        sparse.save_npz(train_matrix_path, train_matrix)
        saved_count += 1
    else:
        logger.info("6️⃣ Train matrix already exists, skipping...")
        skipped_count += 1
    
    if not check_artifact_exists(artifacts_base, 'interaction_matrix.npz') or force_recreate:
        logger.info("   Saving interaction matrix...")
        sparse.save_npz(interaction_matrix_path, interaction_matrix)
        saved_count += 1
    else:
        logger.info("   Interaction matrix already exists, skipping...")
        skipped_count += 1
    
    # 7. Save warm user info
    warm_user_path = artifacts_base / 'warm_user_info.json'
    if not check_artifact_exists(artifacts_base, 'warm_user_info.json') or force_recreate:
        logger.info("7️⃣ Saving warm user info...")
        warm_user_info_serializable = {
            'warm_user_list': [int(u) for u in warm_user_info['warm_user_list']],
            'warm_user_to_matrix_idx': {str(k): int(v) for k, v in warm_user_info['warm_user_to_matrix_idx'].items()}
        }
        with open(warm_user_path, 'w') as f:
            json.dump(warm_user_info_serializable, f, indent=2)
        saved_count += 1
    else:
        logger.info("7️⃣ Warm user info already exists, skipping...")
        skipped_count += 1
    
    # 8. Save product-to-users lookup (EXACTLY like notebook: product_to_users_list)
    lookup_path = artifacts_base / 'product_to_users_lookup.json'
    if not check_artifact_exists(artifacts_base, 'product_to_users_lookup.json') or force_recreate:
        logger.info("8️⃣ Saving product-user lookup...")
        product_to_users_serializable = {
            str(k): [(int(u), float(s)) for u, s in v]
            for k, v in product_to_users_lookup.items()
        }
        with open(lookup_path, 'w') as f:
            json.dump(product_to_users_serializable, f)
        saved_count += 1
    else:
        logger.info("8️⃣ Product-user lookup already exists, skipping...")
        skipped_count += 1
    
    # 9. Save product IDs list
    product_ids_path = artifacts_base / 'product_ids_list.json'
    if not check_artifact_exists(artifacts_base, 'product_ids_list.json') or force_recreate:
        logger.info("9️⃣ Saving product IDs list...")
        with open(product_ids_path, 'w') as f:
            json.dump([str(pid) for pid in product_ids_list], f)
        saved_count += 1
    else:
        logger.info("9️⃣ Product IDs list already exists, skipping...")
        skipped_count += 1
    
    # 10. Save configuration (EXACTLY like notebook)
    config_path = artifacts_base / 'config.json'
    if not check_artifact_exists(artifacts_base, 'config.json') or force_recreate:
        logger.info("🔟 Saving configuration...")
        config_to_save = {
            'event_weights': config.get('event_weights', {}),
            'embedding_dimension': int(all_embeddings.shape[1]),
            'n_components_svd': int(svd_transformer.n_components),
            'explained_variance': float(svd_transformer.explained_variance_ratio_.sum()),
            'faiss_index_type': 'IndexFlatL2',
            'model_hyperparameters': {
                'als_factors': config.get('model', {}).get('als', {}).get('factors', 64),
                'als_regularization': config.get('model', {}).get('als', {}).get('regularization', 0.3),
                'als_iterations': config.get('model', {}).get('als', {}).get('iterations', 30)
            },
            'data_stats': {
                'total_interactions': int(len(df)),
                'total_users': int(mappings['n_users']),
                'total_products': int(mappings['n_products']),
                'warm_users': int(len(warm_user_info['warm_user_list'])),
                'cold_users': int(mappings['n_users'] - len(warm_user_info['warm_user_list']))
            }
        }
        with open(config_path, 'w') as f:
            json.dump(config_to_save, f, indent=2)
        saved_count += 1
    else:
        logger.info("🔟 Configuration already exists, skipping...")
        skipped_count += 1
    
    logger.info(f"✅ Artifact saving complete! Saved: {saved_count}, Skipped: {skipped_count}")

if __name__ == "__main__":
    run_training_pipeline()

