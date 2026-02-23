"""
Baseline (Freq-Recency) hyperparameter tuning.
"""

import numpy as np
import pandas as pd
from itertools import product

from tuning.eval import evaluate_recommendations


def _build_scores(train, recency_weight, quantity_weight):
    """Build per-customer-SKU scores with given weights."""
    max_date = train['INVOICE_DATE'].max()
    min_date = train['INVOICE_DATE'].min()
    max_days = (max_date - min_date).days

    features = (train.groupby(['ACCOUNT_ID', 'SKU_ID'])
                .agg(
                    frequency=('ORDER_ID', 'nunique'),
                    total_quantity=('ITEMS_PHYS_CASES', 'sum'),
                    last_purchase=('INVOICE_DATE', 'max')
                )
                .reset_index())

    features['recency_days'] = (max_date - features['last_purchase']).dt.days
    features['recency_score'] = 1 - (features['recency_days'] / max(max_days, 1))

    features['score'] = (
        features['frequency'] +
        recency_weight * features['recency_score'] +
        quantity_weight * np.log1p(features['total_quantity'])
    )

    return features


def _make_recommend_fn(features, global_pop):
    """
    Return a recommend function that uses customer history + global fallback.
    """
    def recommend(customer_id, k):
        cust = features[features['ACCOUNT_ID'] == customer_id]
        recs = cust.nlargest(k, 'score')['SKU_ID'].tolist()
        
        if len(recs) < k:
            seen = set(recs)
            for sku in global_pop:
                if sku not in seen:
                    recs.append(sku)
                    seen.add(sku)
                if len(recs) >= k:
                    break

        return recs[:k]

    return recommend


def tune_baseline(train, ground_truth, global_pop):
    """Grid search over recency_weight and quantity_weight."""
    print("\n--- Baseline Tuning ---")

    recency_weights = [0.3, 0.5, 0.7]
    quantity_weights = [0.1, 0.3, 0.5]
    k = 5
    combos = list(product(recency_weights, quantity_weights))

    print(f"  recency_weight: {recency_weights}")
    print(f"  quantity_weight: {quantity_weights}")
    print(f"  K={k}, {len(combos)} combinations")

    results = []

    for i, (rec_w, qty_w) in enumerate(combos, 1):
        features = _build_scores(train, rec_w, qty_w)
        recommend_fn = _make_recommend_fn(features, global_pop)
        metrics = evaluate_recommendations(ground_truth, recommend_fn, k)

        results.append({
            'recency_weight': rec_w,
            'quantity_weight': qty_w,
            **metrics,
        })

        print(f"  [{i}/{len(combos)}] rec={rec_w} qty={qty_w} "
              f"-> recall={metrics['recall']:.4f} prec={metrics['precision']:.4f} hit={metrics['hit_rate']:.4f}")

    return pd.DataFrame(results)
