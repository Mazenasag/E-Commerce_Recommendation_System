"""
Helper utilities for metrics, plotting, and common operations
"""
import numpy as np
from typing import List, Dict, Set
from collections import defaultdict

def precision_at_k(recommended: List[int], actual: Set[int], k: int) -> float:
    """Calculate Precision@K"""
    rec_set = set(recommended[:k])
    return len(rec_set & actual) / k if k > 0 else 0.0

def recall_at_k(recommended: List[int], actual: Set[int], k: int) -> float:
    """Calculate Recall@K"""
    rec_set = set(recommended[:k])
    return len(rec_set & actual) / len(actual) if len(actual) > 0 else 0.0

def hit_rate_at_k(recommended: List[int], actual: Set[int], k: int) -> float:
    """Calculate Hit Rate@K (1 if any recommended item is in actual, else 0)"""
    rec_set = set(recommended[:k])
    return 1.0 if len(rec_set & actual) > 0 else 0.0

def ndcg_at_k(recommended: List[int], actual: Set[int], k: int) -> float:
    """Calculate NDCG@K"""
    dcg = 0.0
    for i, item in enumerate(recommended[:k]):
        if item in actual:
            dcg += 1.0 / np.log2(i + 2)
    
    idcg = sum(1.0 / np.log2(i + 2) for i in range(min(len(actual), k)))
    return dcg / idcg if idcg > 0 else 0.0

def calculate_metrics(recommended: List[int], actual: Set[int], k_values: List[int] = [5, 10, 20, 50]) -> Dict[str, Dict[int, float]]:
    """
    Calculate all metrics for different K values
    
    Returns:
        Dict with metrics: {'precision': {5: 0.1, 10: 0.2, ...}, ...}
    """
    metrics = {
        'precision': {},
        'recall': {},
        'hit_rate': {},
        'ndcg': {}
    }
    
    for k in k_values:
        metrics['precision'][k] = precision_at_k(recommended, actual, k)
        metrics['recall'][k] = recall_at_k(recommended, actual, k)
        metrics['hit_rate'][k] = hit_rate_at_k(recommended, actual, k)
        metrics['ndcg'][k] = ndcg_at_k(recommended, actual, k)
    
    return metrics

def format_number(num: int) -> str:
    """Format number with commas"""
    return f"{num:,}"

def ensure_dir(path: str):
    """Ensure directory exists"""
    from pathlib import Path
    Path(path).mkdir(parents=True, exist_ok=True)

