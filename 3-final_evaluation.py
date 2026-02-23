"""
Evaluates the three models across two time-based splits and produces production-ready outputs.
"""

import logging
import warnings

import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime

warnings.filterwarnings("ignore", category=UserWarning, module="sklearn")

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

from models import (
    BaselineFreqRecency, NMFModel, EASEModel,
    build_interaction_matrix, build_fallback_popularity,
    get_dynamic_k, _validate_columns, REQUIRED_TRANS_COLS,
)
from test import temporal_split, evaluate_model
from eda.io import load_and_clean, check_data_quality

OUTPUT_DIR = Path('model_outputs')
OUTPUT_DIR.mkdir(exist_ok=True)


def evaluate_expanding_window(trans, attrs):
    """
    Evaluate models on 2 temporal splits (expanding window).
    Split 1: Train(May-Jun) -> Test(Jul)
    Split 2: Train(May-Jul) -> Test(Aug)
    """
    logger.info("Expanding Window Evaluation")

    splits = [
        ('2022-07', 'Split 1: Train(May-Jun) -> Test(Jul)'),
        ('2022-08', 'Split 2: Train(May-Jul) -> Test(Aug)')
    ]

    all_results = {}

    for test_month, description in splits:
        logger.info("%s", description)

        train, test, ground_truth = temporal_split(trans, test_month=test_month)

        X, _, mappings = build_interaction_matrix(train)
        global_pop, channel_pop = build_fallback_popularity(train, attrs)

        baseline = BaselineFreqRecency(recency_weight=0.3, quantity_weight=0.3)
        baseline.fit(train)

        nmf = NMFModel(n_components=20, blend_weight=0.50)
        nmf.fit(X, mappings)

        ease = EASEModel(lambda_reg=50)
        ease.fit(X, mappings)

        split_results = {}
        test_customers = list(ground_truth.keys())
        for model in [baseline, nmf, ease]:
            metrics = evaluate_model(
                model, test_customers, ground_truth, attrs,
                global_pop, channel_pop
            )
            split_results[model.name] = metrics

        all_results[test_month] = split_results

    return all_results


def aggregate_metrics(all_results):
    """Aggregate metrics across splits and select winner by recall."""
    logger.info("Aggregating metrics across splits")

    model_names = list(all_results['2022-07'].keys())
    aggregated = {}

    for model_name in model_names:
        metrics_jul = all_results['2022-07'][model_name]
        metrics_aug = all_results['2022-08'][model_name]

        aggregated[model_name] = {
            'recall_jul': metrics_jul['recall_mean'],
            'recall_aug': metrics_aug['recall_mean'],
            'recall_avg': (metrics_jul['recall_mean'] + metrics_aug['recall_mean']) / 2,
            'precision_jul': metrics_jul['precision_mean'],
            'precision_aug': metrics_aug['precision_mean'],
            'precision_avg': (metrics_jul['precision_mean'] + metrics_aug['precision_mean']) / 2,
            'map_jul': metrics_jul['map_mean'],
            'map_aug': metrics_aug['map_mean'],
            'map_avg': (metrics_jul['map_mean'] + metrics_aug['map_mean']) / 2,
            'hit_rate_jul': metrics_jul['hit_rate'],
            'hit_rate_aug': metrics_aug['hit_rate'],
            'hit_rate_avg': (metrics_jul['hit_rate'] + metrics_aug['hit_rate']) / 2,
            'n_evaluated_jul': metrics_jul['n_evaluated'],
            'n_evaluated_aug': metrics_aug['n_evaluated'],
            'k_mean_jul': metrics_jul.get('k_mean', 0),
            'k_mean_aug': metrics_aug.get('k_mean', 0),
        }

    for model_name in model_names:
        m = aggregated[model_name]
        logger.info("  %s: Recall=%.4f/%.4f (avg %.4f), Prec=%.4f/%.4f, MAP=%.4f/%.4f, Hit=%.4f/%.4f",
                     model_name,
                     m['recall_jul'], m['recall_aug'], m['recall_avg'],
                     m['precision_jul'], m['precision_aug'],
                     m['map_jul'], m['map_aug'],
                     m['hit_rate_jul'], m['hit_rate_aug'])

    recall_scores = {name: m['recall_avg'] for name, m in aggregated.items()}
    winner = max(recall_scores, key=recall_scores.get)

    logger.info("  Winner (by avg recall): %s (%.4f)", winner, recall_scores[winner])

    return aggregated, winner, recall_scores


def save_aggregated_report(aggregated, winner, scores, fixed_metrics=None, dynamic_metrics=None):
    """Save final comparison report."""
    output_path = OUTPUT_DIR / 'final_model_comparison.txt'

    with open(output_path, 'w', encoding='utf-8') as f:
        f.write("Comparacion Final de Modelos - Ventana Expandible\n")
        f.write(f"Generado: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Splits: Julio 2022, Agosto 2022\n\n")

        f.write(f"{'Modelo':<30} {'Metrica':<15} {'Julio':<10} {'Agosto':<10} {'Promedio':<10}\n")
        f.write(f"{'-'*75}\n")

        for model_name, m in aggregated.items():
            f.write(f"{model_name:<30} {'Recall':<15} {m['recall_jul']:<10.4f} {m['recall_aug']:<10.4f} {m['recall_avg']:<10.4f}\n")
            f.write(f"{'':30} {'Precision':<15} {m['precision_jul']:<10.4f} {m['precision_aug']:<10.4f} {m['precision_avg']:<10.4f}\n")
            f.write(f"{'':30} {'MAP':<15} {m['map_jul']:<10.4f} {m['map_aug']:<10.4f} {m['map_avg']:<10.4f}\n")
            f.write(f"{'':30} {'Hit Rate':<15} {m['hit_rate_jul']:<10.4f} {m['hit_rate_aug']:<10.4f} {m['hit_rate_avg']:<10.4f}\n")
            f.write(f"{'':30} {'n_evaluados':<15} {m['n_evaluated_jul']:<10} {m['n_evaluated_aug']:<10}\n")
            f.write(f"{'':30} {'K promedio':<15} {m['k_mean_jul']:<10.1f} {m['k_mean_aug']:<10.1f}\n")
            f.write(f"{'-'*75}\n")

        f.write(f"\nGanador (por recall promedio): {winner} ({scores[winner]:.4f})\n")

        if fixed_metrics and dynamic_metrics:
            f.write(f"\n--- H4: K Fijo vs K Dinamico (NMF, Agosto) ---\n\n")
            f.write(f"K Fijo = 5 (media global EDA: 3.0 SKUs/orden x 1.5)\n")
            f.write(f"K Dinamico = por cliente segun SkuDistintosPromediosXOrden (rango 3-12)\n\n")

            f.write(f"{'Metrica':<15} {'K Fijo=5':<15} {'K Dinamico':<15} {'Delta':<15}\n")
            f.write(f"{'-'*60}\n")

            for metric in ['recall', 'precision', 'hit_rate']:
                f_val = fixed_metrics[metric]
                d_val = dynamic_metrics[metric]
                delta = d_val - f_val
                delta_pct = (delta / f_val * 100) if f_val > 0 else 0
                sign = '+' if delta >= 0 else ''
                f.write(f"{metric.capitalize():<15} {f_val:<15.4f} {d_val:<15.4f} {sign}{delta:.4f} ({sign}{delta_pct:.1f}%)\n")

            recall_delta = dynamic_metrics['recall'] - fixed_metrics['recall']
            if recall_delta > 0:
                f.write(f"\nH4 validada: K Dinamico mejoro recall en "
                        f"{recall_delta:.4f} ({recall_delta/fixed_metrics['recall']*100:.1f}%) vs K Fijo=5\n")
            else:
                f.write(f"\nH4 no validada: K Dinamico no mejoro recall vs K Fijo=5\n")

    logger.info("  Saved: %s", output_path)


def evaluate_fixed_vs_dynamic_k(trans, attrs):
    """
    H4 Validation: Compare fixed K (derived from EDA mean) vs dynamic K.
    Fixed K = round(3.0 * 1.5) = 5
    Dynamic K = per-customer based on SkuDistintosPromediosXOrden (range 3-12)
    """
    logger.info("H4: Fixed K vs Dynamic K")

    FIXED_K = 5
    test_month = '2022-08'
    train, test, ground_truth = temporal_split(trans, test_month=test_month)
    test_customers = list(ground_truth.keys())

    X, _, mappings = build_interaction_matrix(train)
    global_pop, channel_pop = build_fallback_popularity(train, attrs)

    nmf = NMFModel(n_components=20, blend_weight=0.50)
    nmf.fit(X, mappings)

    # Fixed K evaluation
    fixed_recalls, fixed_precisions, fixed_hits = [], [], 0
    for customer_id in test_customers:
        recs, _ = nmf.recommend(customer_id, FIXED_K, global_pop, channel_pop, attrs)
        recs = list(dict.fromkeys(recs))
        actual = ground_truth.get(customer_id, set())
        if not actual:
            continue
        predicted_set = set(recs)
        fixed_recalls.append(len(predicted_set & actual) / len(actual))
        fixed_precisions.append(len(predicted_set & actual) / len(recs) if recs else 0)
        if predicted_set & actual:
            fixed_hits += 1

    n_evaluated = len(fixed_recalls)

    fixed_metrics = {
        'recall': np.mean(fixed_recalls),
        'precision': np.mean(fixed_precisions),
        'hit_rate': fixed_hits / n_evaluated if n_evaluated > 0 else 0
    }

    # Dynamic K evaluation
    dynamic_recalls, dynamic_precisions, dynamic_hits = [], [], 0
    k_values_used = []
    for customer_id in test_customers:
        k = get_dynamic_k(customer_id, attrs)
        k_values_used.append(k)
        recs, _ = nmf.recommend(customer_id, k, global_pop, channel_pop, attrs)
        recs = list(dict.fromkeys(recs))
        actual = ground_truth.get(customer_id, set())
        if not actual:
            continue
        predicted_set = set(recs)
        dynamic_recalls.append(len(predicted_set & actual) / len(actual))
        dynamic_precisions.append(len(predicted_set & actual) / len(recs) if recs else 0)
        if predicted_set & actual:
            dynamic_hits += 1

    dynamic_metrics = {
        'recall': np.mean(dynamic_recalls),
        'precision': np.mean(dynamic_precisions),
        'hit_rate': dynamic_hits / n_evaluated if n_evaluated > 0 else 0
    }

    for metric in ['recall', 'precision', 'hit_rate']:
        f_val = fixed_metrics[metric]
        d_val = dynamic_metrics[metric]
        delta = d_val - f_val
        delta_pct = (delta / f_val * 100) if f_val > 0 else 0
        sign = '+' if delta >= 0 else ''
        logger.info("  %s: Fixed=%.4f Dynamic=%.4f %s%.4f (%s%.1f%%)",
                     metric, f_val, d_val, sign, delta, sign, delta_pct)

    logger.info("  Dynamic K: min=%d, max=%d, mean=%.1f, n_evaluated=%d",
                min(k_values_used), max(k_values_used),
                np.mean(k_values_used), n_evaluated)

    recall_delta = dynamic_metrics['recall'] - fixed_metrics['recall']
    if recall_delta > 0:
        logger.info("  H4 validated: Dynamic K improved recall by %.4f (%.1f%%)",
                     recall_delta, recall_delta / fixed_metrics['recall'] * 100)
    else:
        logger.info("  H4 not validated: Dynamic K did not improve recall")

    return fixed_metrics, dynamic_metrics


def generate_production_output(trans, attrs, winner_model_name):
    """Generate production recommendations CSV."""
    logger.info("Generating production output")

    train = trans.copy()
    max_date = train['INVOICE_DATE'].max()
    scoring_date = max_date + pd.Timedelta(days=1)

    logger.info("  Training up to %s, scoring: %s", max_date.date(), scoring_date.date())

    global_pop, channel_pop = build_fallback_popularity(train, attrs)

    if winner_model_name == "Baseline Freq-Recency":
        model = BaselineFreqRecency(recency_weight=0.3, quantity_weight=0.3)
        model.fit(train)
    else:
        X, _, mappings = build_interaction_matrix(train)
        if 'NMF' in winner_model_name:
            model = NMFModel(n_components=20, blend_weight=0.50)
        else:
            model = EASEModel(lambda_reg=50)
        model.fit(X, mappings)

    all_customers = attrs['ACCOUNT_ID'].unique()
    rows = []
    for customer_id in all_customers:
        k = get_dynamic_k(customer_id, attrs)
        recs, rec_scores = model.recommend(customer_id, k, global_pop, channel_pop, attrs)

        for rank, (sku, score) in enumerate(zip(recs, rec_scores), 1):
            rows.append({
                'scoring_date': scoring_date.date(),
                'ACCOUNT_ID': customer_id,
                'SKU_ID': sku,
                'rank': rank,
                'score': round(score, 4),
                'k_recommended': k,
                'model': winner_model_name.lower().replace(' ', '_'),
                'generation_ts': datetime.utcnow().isoformat() + 'Z'
            })

    df = pd.DataFrame(rows)
    df = df.sort_values(['ACCOUNT_ID', 'rank']).reset_index(drop=True)

    output_path = OUTPUT_DIR / 'recommendations_production.csv'
    df.to_csv(output_path, index=False)

    logger.info("  %s recommendations for %s customers",
                f"{len(df):,}", f"{len(all_customers):,}")
    logger.info("  K range: %d-%d (mean: %.1f)",
                df['k_recommended'].min(), df['k_recommended'].max(),
                df['k_recommended'].mean())
    logger.info("  Saved: %s", output_path)

    return df


if __name__ == '__main__':
    logger.info("Final Evaluation - Expanding Window")

    trans, attrs = load_and_clean()
    _validate_columns(trans, REQUIRED_TRANS_COLS, "trans")
    trans, attrs, _ = check_data_quality(trans, attrs)

    all_results = evaluate_expanding_window(trans, attrs)
    aggregated, winner, recall_scores = aggregate_metrics(all_results)

    fixed_metrics, dynamic_metrics = evaluate_fixed_vs_dynamic_k(trans, attrs)
    save_aggregated_report(aggregated, winner, recall_scores, fixed_metrics, dynamic_metrics)

    df_output = generate_production_output(trans, attrs, winner)

    logger.info("Done. Winner: %s (avg recall=%.4f)", winner, recall_scores[winner])
    logger.info("Files: final_model_comparison.txt, recommendations_production.csv (%s rows)",
                f"{len(df_output):,}")
