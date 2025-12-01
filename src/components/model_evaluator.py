"""
Model Evaluator Component
Evaluates model performance using various metrics
"""
import numpy as np
from typing import Dict, List, Set, Callable
from collections import defaultdict
from tqdm import tqdm
import sys
from src.utils.logger import get_logger
from src.utils.exception import CustomException
from src.utils.helpers import calculate_metrics

logger = get_logger(__name__)

class ModelEvaluator:
    """Handles model evaluation"""
    
    def __init__(self, config: dict):
        """
        Initialize model evaluator
        
        Args:
            config: Configuration dictionary
        """
        self.config = config
        self.k_values = [5, 10, 20, 50]
    
    def evaluate_recommendations(self, 
                                 get_recommendations_fn: Callable,
                                 test_ground_truth: Dict[int, Set[int]],
                                 sample_size: int = 500) -> Dict:
        """
        Evaluate recommendation function
        
        Args:
            get_recommendations_fn: Function that takes product_idx and returns list of user_idx
            test_ground_truth: Dict mapping product_idx to set of actual user_idx
            sample_size: Number of products to evaluate
        
        Returns:
            Dictionary with evaluation metrics
        """
        try:
            logger.info(f"🔄 Evaluating on {sample_size} products...")
            
            # Sample products
            valid_products = list(test_ground_truth.keys())
            if len(valid_products) > sample_size:
                sample_products = np.random.choice(valid_products, sample_size, replace=False)
            else:
                sample_products = valid_products
            
            # Initialize results
            results = {k: {'precision': [], 'recall': [], 'hit_rate': [], 'ndcg': []} 
                      for k in self.k_values}
            
            # Evaluate each product
            for product_idx in tqdm(sample_products, desc="Evaluating"):
                try:
                    recommended = get_recommendations_fn(product_idx, n=max(self.k_values))
                    actual = test_ground_truth[product_idx]
                    
                    # Calculate metrics for each K
                    metrics = calculate_metrics(recommended, actual, self.k_values)
                    
                    for k in self.k_values:
                        results[k]['precision'].append(metrics['precision'][k])
                        results[k]['recall'].append(metrics['recall'][k])
                        results[k]['hit_rate'].append(metrics['hit_rate'][k])
                        results[k]['ndcg'].append(metrics['ndcg'][k])
                        
                except Exception as e:
                    logger.warning(f"Error evaluating product {product_idx}: {e}")
                    continue
            
            # Calculate averages
            summary = {}
            for k in self.k_values:
                summary[k] = {
                    'precision': np.mean(results[k]['precision']),
                    'recall': np.mean(results[k]['recall']),
                    'hit_rate': np.mean(results[k]['hit_rate']),
                    'ndcg': np.mean(results[k]['ndcg'])
                }
            
            logger.info("✅ Evaluation complete")
            return summary
            
        except Exception as e:
            logger.error(f"Error in evaluation: {e}")
            raise CustomException(f"Failed to evaluate model: {str(e)}", sys)
    
    def print_evaluation_results(self, results: Dict):
        """Print evaluation results in a formatted way"""
        logger.info("\n" + "=" * 70)
        logger.info("📊 EVALUATION RESULTS")
        logger.info("=" * 70)
        
        for k in self.k_values:
            if k in results:
                logger.info(f"\n🎯 Results @ K={k}:")
                logger.info(f"   Precision: {results[k]['precision']:.4f}")
                logger.info(f"   Recall: {results[k]['recall']:.4f}")
                logger.info(f"   Hit Rate: {results[k]['hit_rate']:.4f}")
                logger.info(f"   NDCG: {results[k]['ndcg']:.4f}")

