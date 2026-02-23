"""
Evaluation and data splitting for hyperparameter tuning.
"""

import numpy as np


def temporal_split_validation(trans, train_end='2022-06', val_month='2022-07'):
    """Split for tuning: train on earlier months, validate on target month."""
    trans['year_month'] = trans['INVOICE_DATE'].dt.to_period('M').astype(str)

    train = trans[trans['year_month'] <= train_end].copy()
    val = trans[trans['year_month'] == val_month].copy()

    ground_truth = (val.groupby('ACCOUNT_ID')['SKU_ID']
                    .apply(set)
                    .to_dict())

    return train, val, ground_truth


def evaluate_recommendations(ground_truth, recommend_fn, k=10):
    """
    Evaluate recommendations using a single interface.
    """
    recalls = []
    precisions = []
    hits = 0

    for customer_id, actual in ground_truth.items():
        recs = recommend_fn(customer_id, k)
        predicted = set(recs)

        if len(actual) > 0:
            recalls.append(len(predicted & actual) / len(actual))

        if len(predicted) > 0:
            precisions.append(len(predicted & actual) / len(predicted))

        if predicted & actual:
            hits += 1

    n_evaluated = len(recalls)

    return {
        'recall': np.mean(recalls) if recalls else 0,
        'precision': np.mean(precisions) if precisions else 0,
        'hit_rate': hits / n_evaluated if n_evaluated > 0 else 0,
        'n_evaluated': n_evaluated,
    }
