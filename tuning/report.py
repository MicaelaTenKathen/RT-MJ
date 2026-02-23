"""
Tuning report.
"""

from datetime import datetime


def save_csv(df, path, sort_by='recall'):
    """Save results DataFrame as CSV, sorted by primary metric."""
    df.sort_values(sort_by, ascending=False).to_csv(path, index=False)
    print(f"  Saved: {path}")


def save_consolidated(results_baseline, results_nmf, results_ease, output_dir):
    """
    One short summary TXT with best params per model + comparison table.
    """
    best_b = results_baseline.sort_values('recall', ascending=False).iloc[0]
    best_n = results_nmf.sort_values('recall', ascending=False).iloc[0]
    best_e = results_ease.sort_values('recall', ascending=False).iloc[0]

    path = output_dir / 'tuning_consolidado.txt'

    with open(path, 'w', encoding='utf-8') as f:
        f.write("AJUSTE DE HIPERPARAMETROS - RESUMEN\n")
        f.write(f"Generado: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Train: Mayo-Junio 2022 | Validacion: Julio 2022 | K=5\n\n")

        # Comparison table
        f.write(f"{'Modelo':<25} {'Recall':<10} {'Precision':<12} {'Hit Rate':<10}\n")
        f.write(f"{'-'*57}\n")

        models = [
            ('Baseline Freq-Recency', best_b),
            ('NMF', best_n),
            ('EASE', best_e),
        ]

        best_recall = max(best_b['recall'], best_n['recall'], best_e['recall'])
        for name, row in models:
            marker = ' *' if row['recall'] == best_recall else ''
            f.write(f"{name:<25} {row['recall']:<10.4f} {row['precision']:<12.4f} {row['hit_rate']:<10.4f}{marker}\n")

        f.write(f"\n* = mejor recall\n\n")

        # Best params
        f.write("Mejores parametros:\n\n")

        f.write(f"  Baseline:\n")
        f.write(f"    recency_weight:  {best_b['recency_weight']:.1f}\n")
        f.write(f"    quantity_weight: {best_b['quantity_weight']:.1f}\n\n")

        f.write(f"  NMF:\n")
        f.write(f"    n_components:    {best_n['n_components']:.0f}\n")
        f.write(f"    blend_weight:    {best_n['blend_weight']:.2f}\n\n")

        f.write(f"  EASE:\n")
        f.write(f"    lambda_reg:      {best_e['lambda_reg']:.0f}\n\n")

        # Grid sizes
        total = len(results_baseline) + len(results_nmf) + len(results_ease)
        f.write(f"Evaluaciones: {len(results_baseline)} Baseline + {len(results_nmf)} NMF + {len(results_ease)} EASE = {total} total\n\n")

        f.write("Archivos detallados:\n")
        f.write("  - model_outputs/baseline_tuning.csv\n")
        f.write("  - model_outputs/nmf_tuning.csv\n")
        f.write("  - model_outputs/ease_tuning.csv\n")

    print(f"  Saved: {path}")
