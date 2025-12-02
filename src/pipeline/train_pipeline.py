import pandas as pd
import numpy as np
import pickle
import json
import faiss
from pathlib import Path
from scipy import sparse
from src.components.data_ingestion import DataIngestion
from src.components.data_transformation import DataTransformation
from src.components.model_trainer import ModelTrainer
from src.config_loader import load_config
from src.utils.logger import get_logger
from src.utils.helpers import ensure_dir

logger = get_logger(__name__)

def run_training_pipeline(config_path: str = "config/config.yaml"):
    config = load_config(config_path)
    artifacts_base = Path(config.get('artifacts', {}).get('base_path', 'artifacts'))
    ensure_dir(artifacts_base)
    
    logger.info("Starting training pipeline...")
    
    data_ingestion = DataIngestion(config)
    df = data_ingestion.load_data()
    data_ingestion.validate_data(df)
    
    data_transformation = DataTransformation(config)
    processed_path = config.get('data', {}).get('processed_data_path', 'data/processed/processed_data.csv')
    csv_path = processed_path.replace('.parquet', '.csv') if processed_path.endswith('.parquet') else processed_path + '.csv' if not processed_path.endswith('.csv') else processed_path
    
    if Path(csv_path).exists():
        logger.info("Loading preprocessed data...")
        df = data_transformation.load_processed_data(processed_path)
        mappings = data_transformation.create_id_mappings(df)
        warm_users, cold_users = data_transformation.identify_user_segments(df)
    else:
        logger.info("Preprocessing data...")
        df = data_transformation.clean_data(df)
        df = data_transformation.process_product_names(df)
        df = data_transformation.create_weighted_scores(df)
        mappings = data_transformation.create_id_mappings(df)
        warm_users, cold_users = data_transformation.identify_user_segments(df)
        data_transformation.save_processed_data(df, processed_path)
    
    train_ratio = config.get('data', {}).get('train_test_split_ratio', 0.8)
    date_min = df['event_date'].min()
    date_max = df['event_date'].max()
    split_date = date_min + pd.Timedelta(days=int((date_max - date_min).days * train_ratio))
    
    train_df = df[df['event_date'] < split_date].copy()
    test_df = df[df['event_date'] >= split_date].copy()
    
    logger.info(f"Train/Test split: {len(train_df):,} / {len(test_df):,}")
    
    logger.info("Creating product embeddings...")
    model_trainer = ModelTrainer(config)
    product_info = df[['product_id', 'cleaned_text']].drop_duplicates(subset=['product_id'])
    product_embeddings, tfidf_vectorizer, svd_transformer, product_info_final = model_trainer.create_product_embeddings(product_info)
    
    product_id_to_embedding = dict(zip(product_info_final['product_id'], product_embeddings))
    product_ids_list = list(product_id_to_embedding.keys())
    product_idx_to_id = {i: pid for i, pid in enumerate(product_ids_list)}
    product_id_to_idx_embeddings = {pid: i for i, pid in enumerate(product_ids_list)}
    all_embeddings = np.vstack([product_id_to_embedding[pid] for pid in product_ids_list])
    products_with_embeddings = set(product_id_to_idx_embeddings.keys())
    
    logger.info(f"Products with embeddings: {len(products_with_embeddings):,}")
    logger.info("Building FAISS index...")
    faiss_index = model_trainer.build_faiss_index(all_embeddings)
    
    logger.info("Building interaction matrices...")
    interaction_matrix, _ = model_trainer.build_interaction_matrix(train_df, mappings)
    
    warm_user_indices = set([mappings['user_id_to_idx'][uid] for uid in warm_users if uid in mappings['user_id_to_idx']])
    train_warm = train_df[train_df['user_idx'].isin(warm_user_indices)].copy()
    train_matrix, warm_user_info = model_trainer.build_interaction_matrix(train_warm, mappings, user_segment=warm_users)
    
    logger.info("Training ALS model...")
    als_model = model_trainer.train_als_model(train_matrix)
    
    logger.info("Building product-user lookup...")
    product_to_users_lookup = model_trainer.build_product_user_lookup(interaction_matrix, mappings, products_with_embeddings)
    
    logger.info("Saving artifacts...")
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
    
    logger.info("Training pipeline complete!")
    
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

def save_artifacts(als_model, faiss_index, all_embeddings, tfidf_vectorizer,
                   svd_transformer, mappings, train_matrix, interaction_matrix,
                   warm_user_info, product_to_users_lookup, product_ids_list,
                   product_idx_to_id, df, config, artifacts_base: Path):
    with open(artifacts_base / 'als_model.pkl', 'wb') as f:
        pickle.dump(als_model, f)
    
    faiss.write_index(faiss_index, str(artifacts_base / 'faiss_index.bin'))
    np.save(artifacts_base / 'product_embeddings.npy', all_embeddings)
    
    with open(artifacts_base / 'tfidf_vectorizer.pkl', 'wb') as f:
        pickle.dump(tfidf_vectorizer, f)
    with open(artifacts_base / 'svd_transformer.pkl', 'wb') as f:
        pickle.dump(svd_transformer, f)
    
    mappings_serializable = {
        'user_id_to_idx': {str(k): int(v) for k, v in mappings['user_id_to_idx'].items()},
        'idx_to_user_id': {str(k): int(v) for k, v in mappings['idx_to_user_id'].items()},
        'product_id_to_idx': {str(k): int(v) for k, v in mappings['product_id_to_idx'].items()},
        'idx_to_product_id': {str(k): int(v) for k, v in mappings['idx_to_product_id'].items()},
        'product_idx_to_id': {str(k): int(v) for k, v in product_idx_to_id.items()},
        'n_users': int(mappings['n_users']),
        'n_products': int(mappings['n_products'])
    }
    with open(artifacts_base / 'id_mappings.json', 'w') as f:
        json.dump(mappings_serializable, f, indent=2)
    
    sparse.save_npz(artifacts_base / 'train_matrix.npz', train_matrix)
    sparse.save_npz(artifacts_base / 'interaction_matrix.npz', interaction_matrix)
    
    warm_user_info_serializable = {
        'warm_user_list': [int(u) for u in warm_user_info['warm_user_list']],
        'warm_user_to_matrix_idx': {str(k): int(v) for k, v in warm_user_info['warm_user_to_matrix_idx'].items()}
    }
    with open(artifacts_base / 'warm_user_info.json', 'w') as f:
        json.dump(warm_user_info_serializable, f, indent=2)
    
    product_to_users_serializable = {
        str(k): [(int(u), float(s)) for u, s in v]
        for k, v in product_to_users_lookup.items()
    }
    with open(artifacts_base / 'product_to_users_lookup.json', 'w') as f:
        json.dump(product_to_users_serializable, f)
    
    with open(artifacts_base / 'product_ids_list.json', 'w') as f:
        json.dump([str(pid) for pid in product_ids_list], f)
    
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
    with open(artifacts_base / 'config.json', 'w') as f:
        json.dump(config_to_save, f, indent=2)
    
    logger.info("Artifacts saved")

if __name__ == "__main__":
    run_training_pipeline()

