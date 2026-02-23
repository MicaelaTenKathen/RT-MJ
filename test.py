"""
Evaluation utilities — temporal splits, metrics, comparison reports.
"""

import logging

import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime

from models import (
    BaselineFreqRecency, NMFModel, EASEModel,
    build_interaction_matrix, build_fallback_popularity,
    get_dynamic_k, _validate_columns, REQUIRED_TRANS_COLS,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

OUTPUT_DIR = Path('model_outputs')
OUTPUT_DIR.mkdir(exist_ok=True)


def temporal_split(trans, test_month='2022-08'):
    """Split data temporally. Does NOT mutate the input DataFrame."""
    logger.info("Temporal Split (test=%s)", test_month)

    trans = trans.copy()
    trans['year_month'] = trans['INVOICE_DATE'].dt.to_period('M').astype(str)

    train = trans[trans['year_month'] < test_month].copy()
    test = trans[trans['year_month'] == test_month].copy()

    logger.info("  Train: %s-%s | %s trans, %s customers",
                train['year_month'].min(), train['year_month'].max(),
                f"{len(train):,}", f"{train['ACCOUNT_ID'].nunique():,}")
    logger.info("  Test: %s | %s trans, %s customers",
                test_month, f"{len(test):,}", f"{test['ACCOUNT_ID'].nunique():,}")

    ground_truth = (test.groupby('ACCOUNT_ID')['SKU_ID']
                    .apply(set)
                    .to_dict())

    logger.info("  Ground truth: %s customers", f"{len(ground_truth):,}")

    return train, test, ground_truth


def _average_precision_at_k(predicted, actual, k):
    """
    AP@K = (1 / min(|actual|, K)) * sum_{i=1}^{K} P(i) * rel(i)
    """
    hits = 0
    sum_precision = 0.0

    for i, item in enumerate(predicted[:k], 1):
        if item in actual:
            hits += 1
            sum_precision += hits / i

    denominator = min(len(actual), k)
    return sum_precision / denominator if denominator > 0 else 0.0


def evaluate_model(model, test_customers, ground_truth, attrs,
                   global_pop, channel_pop):
    """Recall@K, Precision@K, MAP@K, Hit Rate. K is dynamic per customer."""
    logger.info("  Evaluating %s...", model.name)

    recalls, precisions, aps = [], [], []
    hits = 0
    k_values = []

    for customer_id in test_customers:
        actual = ground_truth.get(customer_id, set())
        if not actual:
            continue

        k = get_dynamic_k(customer_id, attrs)
        k_values.append(k)
        recs, _ = model.recommend(customer_id, k, global_pop, channel_pop, attrs)
        recs = list(dict.fromkeys(recs))  # dedup preserving order
        predicted_set = set(recs)

        recalls.append(len(predicted_set & actual) / len(actual))

        if recs:
            precisions.append(len(predicted_set & actual) / len(recs))
            aps.append(_average_precision_at_k(recs, actual, k))

        if predicted_set & actual:
            hits += 1

    n_evaluated = len(recalls)

    metrics = {
        'recall_mean': np.mean(recalls) if recalls else 0,
        'precision_mean': np.mean(precisions) if precisions else 0,
        'map_mean': np.mean(aps) if aps else 0,
        'hit_rate': hits / n_evaluated if n_evaluated > 0 else 0,
        'n_evaluated': n_evaluated,
        'k_mean': np.mean(k_values) if k_values else 0,
        'k_min': min(k_values) if k_values else 0,
        'k_max': max(k_values) if k_values else 0,
    }

    logger.info("    Recall=%.4f  Prec=%.4f  MAP=%.4f  Hit=%.4f  (n=%d, K=%.1f avg)",
                metrics['recall_mean'], metrics['precision_mean'],
                metrics['map_mean'], metrics['hit_rate'],
                n_evaluated, metrics['k_mean'])

    return metrics


def compare_models(train, test, ground_truth, attrs):
    """Run and compare all 3 models. Winner = best recall (primary for reorder)."""
    logger.info("Model Comparison")

    global_pop, channel_pop = build_fallback_popularity(train, attrs)

    baseline = BaselineFreqRecency()
    baseline.fit(train)
    baseline_metrics = evaluate_model(baseline, list(ground_truth.keys()), ground_truth,
                                      attrs, global_pop, channel_pop)

    X, _, mappings = build_interaction_matrix(train)

    nmf = NMFModel(n_components=20, blend_weight=0.50)
    nmf.fit(X, mappings)
    nmf_metrics = evaluate_model(nmf, list(ground_truth.keys()), ground_truth,
                                 attrs, global_pop, channel_pop)

    ease = EASEModel(lambda_reg=50)
    ease.fit(X, mappings)
    ease_metrics = evaluate_model(ease, list(ground_truth.keys()), ground_truth,
                                  attrs, global_pop, channel_pop)

    all_metrics = {
        'Baseline Freq-Recency': baseline_metrics,
        nmf.name: nmf_metrics,
        ease.name: ease_metrics,
    }

    recall_scores = {name: m['recall_mean'] for name, m in all_metrics.items()}
    winner = max(recall_scores, key=recall_scores.get)

    logger.info("  Winner (by recall): %s (%.4f)", winner, recall_scores[winner])

    return all_metrics, winner, recall_scores


def save_comparison(all_metrics, winner, recall_scores):
    """Save comparison results to file."""
    output_path = OUTPUT_DIR / 'model_comparison_august.txt'

    with open(output_path, 'w', encoding='utf-8') as f:
        f.write("Model Comparison - August Test\n\n")

        f.write(f"{'Model':<30} {'Recall':<10} {'Prec':<10} {'MAP':<10} "
                f"{'Hit Rate':<10} {'n':<8} {'K avg':<8}\n")
        f.write(f"{'-'*86}\n")

        for model_name, m in all_metrics.items():
            f.write(f"{model_name:<30} "
                    f"{m['recall_mean']:<10.4f} "
                    f"{m['precision_mean']:<10.4f} "
                    f"{m['map_mean']:<10.4f} "
                    f"{m['hit_rate']:<10.4f} "
                    f"{m['n_evaluated']:<8} "
                    f"{m['k_mean']:<8.1f}\n")

        f.write(f"\nWinner (by recall): {winner} ({recall_scores[winner]:.4f})\n")

    logger.info("  Saved: %s", output_path)


if __name__ == '__main__':
    from eda.io import load_and_clean, check_data_quality

    trans, attrs = load_and_clean()
    _validate_columns(trans, REQUIRED_TRANS_COLS, "trans")
    trans, attrs, _ = check_data_quality(trans, attrs)

    train, test, ground_truth = temporal_split(trans, test_month='2022-08')

    all_metrics, winner, recall_scores = compare_models(train, test, ground_truth, attrs)
    save_comparison(all_metrics, winner, recall_scores)

    logger.info("Done. Winner: %s (recall=%.4f)", winner, recall_scores[winner])
