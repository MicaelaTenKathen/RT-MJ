"""
Statistical and business analysis functions for the EDA.
"""

import pandas as pd


def stats(trans):
    """
    Numbers to understand the business.
    """
    print("\n--- Statistics ---")

    n_orders = trans['ORDER_ID'].nunique()
    n_customers = trans['ACCOUNT_ID'].nunique()
    n_skus = trans['SKU_ID'].nunique()
    n_transactions = len(trans)

    print(f"\n-Orders:       {n_orders:>10,}")
    print(f"-Customers:    {n_customers:>10,}")
    print(f"-SKUs:         {n_skus:>10,}")
    print(f"-Transactions: {n_transactions:>10,}")

    orders_per_customer = trans.groupby('ACCOUNT_ID')['ORDER_ID'].nunique()
    print(f"\n-Orders per customer:")
    print(f"  Mean:   {orders_per_customer.mean():.1f}")
    print(f"  Median: {orders_per_customer.median():.0f}")
    print(f"  Min:    {orders_per_customer.min()}")
    print(f"  Max:    {orders_per_customer.max()}")

    items_per_order = trans.groupby('ORDER_ID')['SKU_ID'].nunique()
    print(f"\n-Unique SKUs per order:")
    print(f"  Mean:   {items_per_order.mean():.1f}")
    print(f"  Median: {items_per_order.median():.0f}")
    print(f"  Min:    {items_per_order.min()}")
    print(f"  Max:    {items_per_order.max()}")

    n_interactions = trans.groupby(['ACCOUNT_ID', 'SKU_ID']).ngroups
    sparsity = 1 - (n_interactions / (n_customers * n_skus))
    print(f"\n-Matrix sparsity: {sparsity:.2%}")
    print(f"Found {n_interactions} unique customer-SKU pairs out of {n_customers*n_skus} possible.")


def analyze_product_popularity(trans):
    """
    Volume (quantity sold) vs how many customers buy it.
    """
    print("\n--- Popularity Analysis ---")

    qty_by_sku = (trans.groupby('SKU_ID')['ITEMS_PHYS_CASES']
                  .sum()
                  .sort_values(ascending=False))

    orders_by_sku = (trans.groupby('SKU_ID')['ORDER_ID']
                     .nunique()
                     .sort_values(ascending=False))

    customers_by_sku = (trans.groupby('SKU_ID')['ACCOUNT_ID']
                        .nunique()
                        .sort_values(ascending=False))

    print("\n-Top 10 SKUs by total quantity sold:")
    for i, (sku, qty) in enumerate(qty_by_sku.head(10).items(), 1):
        print(f"  {i:2d}. SKU {sku:6d}: {qty:>10,.0f} units")

    print("\n-Top 10 SKUs by number of orders:")
    for i, (sku, n_orders) in enumerate(orders_by_sku.head(10).items(), 1):
        n_customers = customers_by_sku[sku]
        print(f"  {i:2d}. SKU {sku:6d}: {n_orders:>6,} orders from {n_customers:>5,} customers")

    total_orders = orders_by_sku.sum()
    cumsum = orders_by_sku.cumsum()
    pct_cumsum = cumsum / total_orders * 100
    n_skus_80pct = (pct_cumsum < 80).sum() + 1
    n_skus_95pct = (pct_cumsum < 95).sum() + 1

    print(f"\n-Concentration (Pareto analysis):")
    print(f"  {n_skus_80pct} SKUs ({n_skus_80pct/len(orders_by_sku)*100:.1f}%) account for 80% of orders")
    print(f"  {n_skus_95pct} SKUs ({n_skus_95pct/len(orders_by_sku)*100:.1f}%) account for 95% of orders")

    return orders_by_sku, customers_by_sku


def analyze_repurchase(trans):
    """
    Returns repurchase dataframe.
    """
    print("\n--- Repurchase Behavior ---")

    repurchase = (trans.groupby(['ACCOUNT_ID', 'SKU_ID'])
                  .agg(n_orders=('ORDER_ID', 'nunique'))
                  .reset_index())

    repurchase_rate = (repurchase['n_orders'] >= 2).mean()

    print(f"\n-Repurchase rate: {repurchase_rate:.1%}")
    print(f"{repurchase_rate:.1%} of (customer, SKU) pairs have 2+ orders")

    print(f"\n-Distribution of orders per (customer, SKU) pair:")
    for i in [1, 2, 3, 5, 10]:
        pct = (repurchase['n_orders'] >= i).mean()
        count = (repurchase['n_orders'] >= i).sum()
        print(f"  {i:2d}+ orders: {pct:6.1%} ({count:>7,} pairs)")

    print(f"\n-Distribution by ranges:")
    range_1 = (repurchase['n_orders'] == 1).mean()
    range_2_3 = ((repurchase['n_orders'] >= 2) & (repurchase['n_orders'] <= 3)).mean()
    range_4_5 = ((repurchase['n_orders'] >= 4) & (repurchase['n_orders'] <= 5)).mean()
    range_6_plus = (repurchase['n_orders'] >= 6).mean()

    print(f"  1 orden:     {range_1:6.1%}")
    print(f"  2-3 ordenes: {range_2_3:6.1%}")
    print(f"  4-5 ordenes: {range_4_5:6.1%}")
    print(f"  6+ ordenes:  {range_6_plus:6.1%}")

    return repurchase


def analyze_temporal_patterns(trans):
    """
    Returns monthly dataframe and by_day series.
    """
    print("\n--- Temporal Patterns ---")

    trans = trans.copy()
    trans['year_month'] = trans['INVOICE_DATE'].dt.to_period('M').astype(str)
    trans['day_of_week'] = trans['INVOICE_DATE'].dt.day_name()
    trans['is_weekend'] = trans['INVOICE_DATE'].dt.dayofweek >= 5

    monthly = (trans.groupby('year_month')
               .agg(n_orders=('ORDER_ID', 'nunique'),
                    n_customers=('ACCOUNT_ID', 'nunique'),
                    n_transactions=('SKU_ID', 'size'),
                    total_qty=('ITEMS_PHYS_CASES', 'sum'))
               .reset_index())

    print("\n-Monthly activity:")
    print(monthly.to_string(index=False))

    may_orders = monthly[monthly['year_month'] == '2022-05']['n_orders'].values[0]
    if may_orders < monthly['n_orders'].median() * 0.5:
        print(f"{may_orders} may orders vs ~{monthly['n_orders'].median():.0f} median)")
        print("May 2022 seems unusually low. Check for data issues.")

    day_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
    by_day = (trans.groupby('day_of_week')
              .agg(n_orders=('ORDER_ID', 'nunique'))
              .reindex(day_order))

    print(f"\n-Orders by day of week:")
    for day, count in by_day['n_orders'].items():
        if pd.notna(count):
            bar = '#' * int(count / by_day['n_orders'].max() * 40)
            print(f"  {day:9s}: {count:>6,.0f} {bar}")
        else:
            print(f"  {day:9s}: No data")

    weekend_pct = trans['is_weekend'].mean()
    print(f"\n-Weekend orders: {weekend_pct:.1%}")

    return monthly, by_day


def _compute_avg_order_size(merged, group_col):
    """
    Compute average unique SKUs per order, grouped by group_col.
    Replaces the inline lambda that was hard to read.
    """
    skus_per_order = merged.groupby([group_col, 'ORDER_ID'])['SKU_ID'].nunique()
    return skus_per_order.groupby(group_col).mean()


def analyze_segments(trans, attrs):
    """
    Returns by_channel, by_segment, crosstab.
    """
    print("\n--- Customer Segment Analysis ---")

    merged = trans.merge(attrs[['ACCOUNT_ID', 'canal', 'segmentoUnico']],
                         on='ACCOUNT_ID', how='left')

    print("\n-Orders by channel:")
    by_channel = (merged.groupby('canal')
                  .agg(n_orders=('ORDER_ID', 'nunique'),
                       n_customers=('ACCOUNT_ID', 'nunique'))
                  .sort_values('n_orders', ascending=False))
    by_channel['avg_order_size'] = _compute_avg_order_size(merged, 'canal')

    print(by_channel)

    missing_channel = merged['canal'].isna().sum()
    if missing_channel > 0:
        print(f"\n {missing_channel:,} transactions ({missing_channel/len(merged)*100:.2f}%) missing channel info")

    print("\n-Orders by segment:")
    by_segment = (merged.groupby('segmentoUnico')
                  .agg(n_orders=('ORDER_ID', 'nunique'),
                       n_customers=('ACCOUNT_ID', 'nunique'))
                  .sort_values('n_orders', ascending=False))
    by_segment['avg_order_size'] = _compute_avg_order_size(merged, 'segmentoUnico')

    print(by_segment)

    crosstab = None
    if merged['segmentoUnico'].notna().any() and merged['canal'].notna().any():
        print("\n-Cross-tab: Segment x Channel (order count):")
        crosstab = pd.crosstab(merged['segmentoUnico'].fillna('Unknown'),
                               merged['canal'].fillna('Unknown'),
                               values=merged['ORDER_ID'],
                               aggfunc='nunique')
        print(crosstab)

    return by_channel, by_segment, crosstab


def analyze_customer_diversity(trans, attrs):
    """
    Returns diversity dataframe.
    """
    print("\n--- Customer Diversity Analysis ---")

    diversity = (trans.groupby('ACCOUNT_ID')
                 .agg(n_unique_skus=('SKU_ID', 'nunique'),
                      n_orders=('ORDER_ID', 'nunique'))
                 .reset_index())

    diversity['avg_skus_per_order'] = diversity['n_unique_skus'] / diversity['n_orders']

    print(f"\n-Unique SKUs purchased per customer:")
    print(f"  Mean:   {diversity['n_unique_skus'].mean():.1f}")
    print(f"  Median: {diversity['n_unique_skus'].median():.0f}")
    print(f"  Min:    {diversity['n_unique_skus'].min()}")
    print(f"  Max:    {diversity['n_unique_skus'].max()}")

    print(f"\n-SKUs per order (per customer):")
    print(f"  Mean:   {diversity['avg_skus_per_order'].mean():.1f}")
    print(f"  Median: {diversity['avg_skus_per_order'].median():.1f}")

    if 'SkuDistintosPromediosXOrden' in attrs.columns:
        merged_div = diversity.merge(attrs[['ACCOUNT_ID', 'SkuDistintosPromediosXOrden']],
                                     on='ACCOUNT_ID', how='left')

        corr = merged_div[['avg_skus_per_order', 'SkuDistintosPromediosXOrden']].corr().iloc[0, 1]
        print(f"\n-Correlation between computed avg and attribute avg (SkuDistintosPromediosXOrden): {corr:.3f}")
        print(f"{'High' if abs(corr) > 0.8 else 'Moderate' if abs(corr) > 0.5 else 'Low'} correlation")

    return diversity


def analyze_attributes(attrs):
    """
    What features do we have? Are they useful?
    """
    print("\n--- Customer Attributes Analysis ---")

    print(f"\n-Total customers in attributes file: {len(attrs):,}")

    for col in attrs.columns:
        if col == 'ACCOUNT_ID':
            continue

        dtype = attrs[col].dtype
        n_missing = attrs[col].isna().sum()
        pct_missing = n_missing / len(attrs) * 100
        n_unique = attrs[col].nunique()

        print(f"\n-{col}:")
        print(f"  Type: {dtype}")
        print(f"  Missing: {n_missing:,} ({pct_missing:.1f}%)")
        print(f"  Unique values: {n_unique:,}")

        if dtype in ['object', 'category'] or n_unique < 20:
            top5 = attrs[col].value_counts().head(5)
            print(f"  Top values:")
            for val, count in top5.items():
                print(f"    {val}: {count:,} ({count/len(attrs)*100:.1f}%)")
        else:
            print(f"  Distribution:")
            print(f"    Mean:   {attrs[col].mean():.2f}")
            print(f"    Median: {attrs[col].median():.2f}")
            print(f"    Min:    {attrs[col].min():.2f}")
            print(f"    Max:    {attrs[col].max():.2f}")
