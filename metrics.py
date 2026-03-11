"""
Retrieval evaluation metrics for the RAG experiment.
Implements: Hit Rate@k, MRR, Precision@k, Recall@k, nDCG@k
"""

import numpy as np
from typing import Dict, List


def hit_rate_at_k(retrieved_keys: List[str], relevant_keys: List[str], k: int) -> float:
    """1.0 if any relevant key is in the top-k retrieved, else 0.0."""
    top_k = retrieved_keys[:k]
    return 1.0 if any(key in relevant_keys for key in top_k) else 0.0


def reciprocal_rank(retrieved_keys: List[str], relevant_keys: List[str]) -> float:
    """1/rank of the first relevant document. 0 if none found."""
    for i, key in enumerate(retrieved_keys):
        if key in relevant_keys:
            return 1.0 / (i + 1)
    return 0.0


def precision_at_k(retrieved_keys: List[str], relevant_keys: List[str], k: int) -> float:
    """Fraction of top-k that are relevant."""
    top_k = retrieved_keys[:k]
    if not top_k:
        return 0.0
    relevant_count = sum(1 for key in top_k if key in relevant_keys)
    return relevant_count / len(top_k)


def recall_at_k(retrieved_keys: List[str], relevant_keys: List[str], k: int) -> float:
    """Fraction of all relevant docs found in top-k."""
    if not relevant_keys:
        return 0.0
    top_k = retrieved_keys[:k]
    found = sum(1 for key in relevant_keys if key in top_k)
    return found / len(relevant_keys)


def ndcg_at_k(retrieved_keys: List[str], relevant_keys: List[str], k: int, primary_key: str = None) -> float:
    """
    Normalized Discounted Cumulative Gain.
    Relevance scores: primary_key=2, other relevant=1, irrelevant=0.
    """
    def relevance(key):
        if primary_key and key == primary_key:
            return 2.0
        if key in relevant_keys:
            return 1.0
        return 0.0

    top_k = retrieved_keys[:k]

    # DCG
    dcg = sum(relevance(key) / np.log2(i + 2) for i, key in enumerate(top_k))

    # Ideal DCG: sort by relevance
    ideal_scores = sorted([relevance(key) for key in relevant_keys] + [0.0] * max(0, k - len(relevant_keys)), reverse=True)[:k]
    # If primary_key exists, make sure score 2.0 is first
    if primary_key:
        ideal_scores = [2.0] + sorted([1.0] * (len(relevant_keys) - 1) + [0.0] * max(0, k - len(relevant_keys)), reverse=True)[:k-1]

    idcg = sum(score / np.log2(i + 2) for i, score in enumerate(ideal_scores))

    if idcg == 0:
        return 0.0
    return dcg / idcg


def evaluate_retrieval(
    retrieved_keys: List[str],
    relevant_keys: List[str],
    primary_key: str = None,
    k_values: List[int] = [1, 3, 5]
) -> Dict:
    """Compute all retrieval metrics for a single query."""
    results = {
        "mrr": reciprocal_rank(retrieved_keys, relevant_keys),
    }
    for k in k_values:
        results[f"hit_rate@{k}"] = hit_rate_at_k(retrieved_keys, relevant_keys, k)
        results[f"precision@{k}"] = precision_at_k(retrieved_keys, relevant_keys, k)
        results[f"recall@{k}"] = recall_at_k(retrieved_keys, relevant_keys, k)
        results[f"ndcg@{k}"] = ndcg_at_k(retrieved_keys, relevant_keys, k, primary_key)
    return results


def aggregate_metrics(all_results: List[Dict]) -> Dict:
    """Average metrics across all queries."""
    if not all_results:
        return {}
    keys = all_results[0].keys()
    return {key: np.mean([r[key] for r in all_results]) for key in keys}
