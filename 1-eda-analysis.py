"""
EDA
"""

from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

from eda.io import load_and_clean, check_data_quality
from eda.quality import (
    stats,
    analyze_product_popularity,
    analyze_repurchase,
    analyze_temporal_patterns,
    analyze_segments,
    analyze_customer_diversity,
    analyze_attributes,
)
from eda.plots import (
    plot_product_popularity,
    plot_repurchase,
    plot_temporal,
    plot_segment_heatmap,
    plot_customer_diversity,
)
from eda.report import save_summary

OUTPUT_DIR = Path('eda_outputs')
OUTPUT_DIR.mkdir(exist_ok=True)


if __name__ == '__main__':
    print("Starting EDA...")

    trans, attrs = load_and_clean()
    trans, attrs, exact_dups_removed = check_data_quality(trans, attrs)

    stats(trans)

    orders_by_sku, _ = analyze_product_popularity(trans)
    plot_product_popularity(orders_by_sku, OUTPUT_DIR)

    repurchase = analyze_repurchase(trans)
    plot_repurchase(repurchase, OUTPUT_DIR)

    monthly, by_day = analyze_temporal_patterns(trans)
    plot_temporal(monthly, by_day, OUTPUT_DIR)

    by_channel, by_segment, crosstab = analyze_segments(trans, attrs)
    plot_segment_heatmap(crosstab, OUTPUT_DIR)

    diversity = analyze_customer_diversity(trans, attrs)
    plot_customer_diversity(diversity, OUTPUT_DIR)

    analyze_attributes(attrs)

    save_summary(trans, attrs, orders_by_sku, repurchase, monthly, diversity,
                 by_channel, by_segment, exact_dups_removed, OUTPUT_DIR)

    print(f"\nEDA complete. Outputs saved to: {OUTPUT_DIR}")