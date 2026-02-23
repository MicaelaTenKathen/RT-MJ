"""
Hyperparameter Tuning - All Models
"""

from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

from eda.io import load_and_clean, check_data_quality
from models import build_interaction_matrix, build_fallback_popularity

from tuning.eval import temporal_split_validation
from tuning.baseline import tune_baseline
from tuning.nmf import tune_nmf
from tuning.ease import tune_ease
from tuning.report import save_csv, save_consolidated

OUTPUT_DIR = Path('model_outputs')
OUTPUT_DIR.mkdir(exist_ok=True)


if __name__ == '__main__':
    print("Hyperparameter Tuning - All Models")

    trans, attrs = load_and_clean()
    trans, attrs, _ = check_data_quality(trans, attrs)

    train, val, ground_truth = temporal_split_validation(trans)

    print(f"\n  Train: {len(train):,} transactions ({train['ACCOUNT_ID'].nunique():,} customers)")
    print(f"  Val:   {len(val):,} transactions ({val['ACCOUNT_ID'].nunique():,} customers)")
    print(f"  Ground truth: {len(ground_truth):,} customers")

    global_pop, channel_pop = build_fallback_popularity(train, attrs)
    X, _, mappings = build_interaction_matrix(train)

    # 1/3 Baseline
    results_baseline = tune_baseline(train, ground_truth, global_pop)
    save_csv(results_baseline, OUTPUT_DIR / 'baseline_tuning.csv')

    # 2/3 NMF
    results_nmf = tune_nmf(ground_truth, attrs, X, mappings, global_pop, channel_pop)
    save_csv(results_nmf, OUTPUT_DIR / 'nmf_tuning.csv')

    # 3/3 EASE
    results_ease = tune_ease(ground_truth, attrs, X, mappings, global_pop, channel_pop)
    save_csv(results_ease, OUTPUT_DIR / 'ease_tuning.csv')

    # Consolidated report
    save_consolidated(results_baseline, results_nmf, results_ease, OUTPUT_DIR)

    # Print summary
    best_b = results_baseline.sort_values('recall', ascending=False).iloc[0]
    best_n = results_nmf.sort_values('recall', ascending=False).iloc[0]
    best_e = results_ease.sort_values('recall', ascending=False).iloc[0]

    print(f"\n--- Best params ---")
    print(f"  Baseline: rec={best_b['recency_weight']:.1f} qty={best_b['quantity_weight']:.1f} (recall={best_b['recall']:.4f})")
    print(f"  NMF:      comp={best_n['n_components']:.0f} blend={best_n['blend_weight']:.2f} (recall={best_n['recall']:.4f})")
    print(f"  EASE:     lambda={best_e['lambda_reg']:.0f} (recall={best_e['recall']:.4f})")
    print(f"\nOutputs saved to: {OUTPUT_DIR}")