"""
EASE hyperparameter tuning.
"""

import pandas as pd

from models import EASEModel
from tuning.eval import evaluate_recommendations


def tune_ease(ground_truth, attrs, X, mappings, global_pop, channel_pop):
    """Grid search over lambda_reg."""
    print("\n--- EASE Tuning ---")

    lambda_values = [50, 100, 150, 200, 300]
    k = 10

    print(f"  lambda_reg: {lambda_values}")
    print(f"  K={k}, {len(lambda_values)} combinations")

    results = []

    for i, lambda_reg in enumerate(lambda_values, 1):
        model = EASEModel(lambda_reg=lambda_reg)
        model.fit(X, mappings)

        recommend_fn = lambda cid, k_, m=model: m.recommend(cid, k_, global_pop, channel_pop, attrs)[0]
        metrics = evaluate_recommendations(ground_truth, recommend_fn, k)

        results.append({
            'lambda_reg': lambda_reg,
            **metrics,
        })

        print(f"  [{i}/{len(lambda_values)}] lambda={lambda_reg} "
              f"-> recall={metrics['recall']:.4f} prec={metrics['precision']:.4f} hit={metrics['hit_rate']:.4f}")

    return pd.DataFrame(results)
