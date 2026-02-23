"""
Plot generation for the EDA.
"""

import matplotlib.pyplot as plt
import seaborn as sns

sns.set_style('whitegrid')
plt.rcParams['figure.figsize'] = (12, 6)


def plot_product_popularity(orders_by_sku, output_dir):
    """Pareto curve + popularity distribution."""
    total_orders = orders_by_sku.sum()
    cumsum = orders_by_sku.cumsum()
    pct_cumsum = cumsum / total_orders * 100
    n_skus_80pct = (pct_cumsum < 80).sum() + 1
    n_skus_95pct = (pct_cumsum < 95).sum() + 1

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    ax1.plot(range(1, len(pct_cumsum)+1), pct_cumsum, linewidth=2)
    ax1.axhline(80, color='red', linestyle='--', alpha=0.7, label=f'80% ({n_skus_80pct} SKUs)')
    ax1.axhline(95, color='orange', linestyle='--', alpha=0.7, label=f'95% ({n_skus_95pct} SKUs)')
    ax1.set_xlabel('Number of SKUs (ranked by order frequency)')
    ax1.set_ylabel('Cumulative % of orders')
    ax1.set_title('Pareto Curve - Product Concentration')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    ax2.hist(orders_by_sku.clip(upper=100), bins=50, edgecolor='white', color='steelblue')
    ax2.set_xlabel('Number of orders per SKU (clipped at 100)')
    ax2.set_ylabel('Number of SKUs')
    ax2.set_title('Distribution of SKU Popularity')
    ax2.axvline(orders_by_sku.median(), color='red', linestyle='--',
                label=f'Median: {orders_by_sku.median():.0f}')
    ax2.legend()

    plt.tight_layout()
    path = output_dir / 'product_popularity.png'
    plt.savefig(path, dpi=150, bbox_inches='tight')
    print(f"  Saved: {path}")
    plt.close()


def plot_repurchase(repurchase, output_dir):
    """Repurchase distribution histogram."""
    repurchase_rate = (repurchase['n_orders'] >= 2).mean()

    fig, ax = plt.subplots(figsize=(10, 5))
    hist_data = repurchase['n_orders'].clip(upper=20)
    ax.hist(hist_data, bins=range(1, 22), edgecolor='white', color='teal')
    ax.set_xlabel('Number of orders per (customer, SKU) pair')
    ax.set_ylabel('Frequency')
    ax.set_title(f'Repurchase Distribution (repurchase rate: {repurchase_rate:.1%})')
    ax.axvline(2, color='red', linestyle='--', alpha=0.7, label='Repurchase threshold (2+ orders)')
    ax.legend()

    plt.tight_layout()
    path = output_dir / 'repurchase_distribution.png'
    plt.savefig(path, dpi=150, bbox_inches='tight')
    print(f"  Saved: {path}")
    plt.close()


def plot_temporal(monthly, by_day, output_dir):
    """Monthly trends + orders by day of week."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    monthly.plot(x='year_month', y=['n_orders', 'n_customers'], marker='o', ax=ax1)
    ax1.set_title('Monthly Trends')
    ax1.set_xlabel('Month')
    ax1.set_ylabel('Count')
    ax1.legend(['Orders', 'Active Customers'])
    ax1.grid(True, alpha=0.3)

    by_day['n_orders'].plot(kind='bar', ax=ax2, color='coral')
    ax2.set_title('Orders by Day of Week')
    ax2.set_xlabel('Day')
    ax2.set_ylabel('Number of Orders')
    ax2.tick_params(axis='x', rotation=45)

    plt.tight_layout()
    path = output_dir / 'temporal_patterns.png'
    plt.savefig(path, dpi=150, bbox_inches='tight')
    print(f"  Saved: {path}")
    plt.close()


def plot_segment_heatmap(crosstab, output_dir):
    """Heatmap of segment x channel order distribution."""
    if crosstab is None:
        return

    fig, ax = plt.subplots(figsize=(12, 6))
    sns.heatmap(crosstab, annot=True, fmt='.0f', cmap='YlOrRd', ax=ax, cbar_kws={'label': 'Orders'})
    ax.set_title('Order Distribution: Segment x Channel')
    ax.set_xlabel('Channel')
    ax.set_ylabel('Segment')

    plt.tight_layout()
    path = output_dir / 'segment_channel_heatmap.png'
    plt.savefig(path, dpi=150, bbox_inches='tight')
    print(f"  Saved: {path}")
    plt.close()


def plot_customer_diversity(diversity, output_dir):
    """Customer diversity histograms."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    ax1.hist(diversity['n_unique_skus'].clip(upper=100), bins=50, edgecolor='white', color='navy')
    ax1.set_xlabel('Unique SKUs per customer')
    ax1.set_ylabel('Number of customers')
    ax1.set_title('Customer Diversity - Total Unique SKUs')
    ax1.axvline(diversity['n_unique_skus'].median(), color='red', linestyle='--',
                label=f"Median: {diversity['n_unique_skus'].median():.0f}")
    ax1.legend()

    ax2.hist(diversity['avg_skus_per_order'].clip(upper=30), bins=50, edgecolor='white', color='darkgreen')
    ax2.set_xlabel('Average SKUs per order')
    ax2.set_ylabel('Number of customers')
    ax2.set_title('Customer Diversity - Avg SKUs per Order')
    ax2.axvline(diversity['avg_skus_per_order'].median(), color='red', linestyle='--',
                label=f"Median: {diversity['avg_skus_per_order'].median():.1f}")
    ax2.legend()

    plt.tight_layout()
    path = output_dir / 'customer_diversity.png'
    plt.savefig(path, dpi=150, bbox_inches='tight')
    print(f"  Saved: {path}")
    plt.close()
