"""
NMF hyperparameter tuning.
"""

import pandas as pd
from itertools import product

from models import NMFModel
from tuning.eval import evaluate_recommendations


def tune_nmf(ground_truth, attrs, X, mappings, global_pop, channel_pop):
    """Grid search over n_components and blend_weight."""
    print("\n--- NMF Tuning ---")

    n_components_values = [20, 30, 40]
    blend_weight_values = [0.2, 0.35, 0.5]
    k = 10
    combos = list(product(n_components_values, blend_weight_values))

    print(f"  n_components: {n_components_values}")
    print(f"  blend_weight: {blend_weight_values}")
    print(f"  K={k}, {len(combos)} combinations")

    results = []

    for i, (n_comp, blend_w) in enumerate(combos, 1):
        model = NMFModel(n_components=n_comp, blend_weight=blend_w)
        model.fit(X, mappings)

        recommend_fn = lambda cid, k_, m=model: m.recommend(cid, k_, global_pop, channel_pop, attrs)[0]
        metrics = evaluate_recommendations(ground_truth, recommend_fn, k)

        results.append({
            'n_components': n_comp,
            'blend_weight': blend_w,
            **metrics,
        })

        print(f"  [{i}/{len(combos)}] comp={n_comp} blend={blend_w} "
              f"-> recall={metrics['recall']:.4f} prec={metrics['precision']:.4f} hit={metrics['hit_rate']:.4f}")

    return pd.DataFrame(results)
