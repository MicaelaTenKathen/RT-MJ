"""
Data loading and quality checks.
"""

import numpy as np
import pandas as pd


def load_and_clean():
    """
    Load both datasets and basic cleanup.
    """
    trans = pd.read_csv('documents/transacciones.csv')
    attrs = pd.read_csv('documents/atributos.csv')

    trans = trans.loc[:, ~trans.columns.str.contains('^Unnamed')]
    attrs = attrs.loc[:, ~attrs.columns.str.contains('^Unnamed')]

    attrs = attrs.rename(columns={'POC': 'ACCOUNT_ID'})

    trans['INVOICE_DATE'] = pd.to_datetime(trans['INVOICE_DATE'].astype(str), format='%Y%m%d')

    trans['ACCOUNT_ID'] = trans['ACCOUNT_ID'].astype(int)
    trans['SKU_ID'] = trans['SKU_ID'].astype(int)
    attrs['ACCOUNT_ID'] = attrs['ACCOUNT_ID'].astype(int)

    print(f"Loaded {len(trans):,} transactions and {len(attrs):,} customer records")

    return trans, attrs


def check_data_quality(trans, attrs):
    """
    Check for nulls, duplicates, inconsistencies.
    """
    print("\n--- Data Quality ---")

    print("\nTRANSACTIONS:")
    print(f"-Shape: {trans.shape}")
    print(f"\n-Null counts:")
    nulls = trans.isnull().sum()
    print(nulls[nulls > 0] if nulls.sum() > 0 else "No nulls found")

    dups = trans.duplicated().sum()
    print(f"\n-Exact duplicate rows: {dups:,}")

    exact_dups_removed = 0

    if dups > 0:
        exact_dup_mask = trans.duplicated(keep=False)
        exact_duplicates = trans[exact_dup_mask].sort_values(['ACCOUNT_ID', 'ORDER_ID', 'SKU_ID', 'INVOICE_DATE', 'ITEMS_PHYS_CASES'])

        print(f"\n  Sample of exact duplicates (all columns identical):")
        print(exact_duplicates[['ACCOUNT_ID', 'ORDER_ID', 'SKU_ID', 'INVOICE_DATE', 'ITEMS_PHYS_CASES']].head(10).to_string(index=False))

        original_count = len(trans)
        trans = trans.drop_duplicates(keep='first')
        rows_removed = original_count - len(trans)
        exact_dups_removed = rows_removed

        print(f"\n   Removed {rows_removed:,} duplicate rows")
        print(f"    Before: {original_count:,} rows -- After: {len(trans):,} rows")

    logical_dups = trans.duplicated(subset=['ACCOUNT_ID', 'ORDER_ID', 'SKU_ID']).sum()
    print(f"\n-Logical duplicates (ACCOUNT_ID + ORDER_ID + SKU_ID): {logical_dups:,}")

    logical_dups = trans.duplicated(subset=['ACCOUNT_ID', 'ORDER_ID', 'SKU_ID', 'ITEMS_PHYS_CASES']).sum()
    print(f"\n-Logical duplicates (ACCOUNT_ID + ORDER_ID + SKU_ID + ITEMS_PHYS_CASES): {logical_dups:,}")

    if logical_dups > 0:

        dup_mask = trans.duplicated(subset=['ACCOUNT_ID', 'ORDER_ID', 'SKU_ID'], keep=False)
        duplicates = trans[dup_mask].sort_values(['ACCOUNT_ID', 'ORDER_ID', 'SKU_ID'])

        print(f"\n  Sample of duplicated (ACCOUNT_ID, ORDER_ID, SKU_ID):")
        print(duplicates[['ACCOUNT_ID', 'ORDER_ID', 'SKU_ID', 'INVOICE_DATE', 'ITEMS_PHYS_CASES']].head(10).to_string(index=False))

        dup_with_dates = trans[dup_mask].sort_values(['ACCOUNT_ID','ORDER_ID', 'SKU_ID', 'INVOICE_DATE'])
        dup_groups = dup_with_dates.groupby(['ACCOUNT_ID','ORDER_ID', 'SKU_ID'])

        date_diffs = []
        for _, group in dup_groups:
            if len(group) > 1:
                dates = group['INVOICE_DATE'].sort_values()
                diff_days = (dates.max() - dates.min()).days
                date_diffs.append(diff_days)

        if date_diffs:
            print(f"\n  Date difference (avg):    {np.mean(date_diffs):.1f} days")
            print(f"\n  Date difference (max):    {np.max(date_diffs)} days")


    print(f"\n-Date range: {trans['INVOICE_DATE'].min().date()} to {trans['INVOICE_DATE'].max().date()}")
    date_diff = (trans['INVOICE_DATE'].max() - trans['INVOICE_DATE'].min()).days
    print(f"   {date_diff} days (~{date_diff/30:.1f} months)")

    neg_qty = (trans['ITEMS_PHYS_CASES'] <= 0).sum()
    print(f"\n-Zero quantities: {neg_qty:,}")
    if neg_qty > 0:
        print("Check for errors")

    print("\n  Attributes:")
    print(f"-Shape: {attrs.shape}")
    print(f"\n-Null counts:")
    nulls_attrs = attrs.isnull().sum()
    print(nulls_attrs[nulls_attrs > 0] if nulls_attrs.sum() > 0 else "No nulls found")

    dup_customers = attrs['ACCOUNT_ID'].duplicated().sum()
    print(f"\n-Duplicate ACCOUNT_IDs: {dup_customers}")

    trans_with_attrs = trans['ACCOUNT_ID'].isin(attrs['ACCOUNT_ID']).sum()
    coverage = trans_with_attrs / len(trans) * 100
    print(f"\n% of transactions with customer attributes: {coverage:.2f}%")

    missing_customers = trans[~trans['ACCOUNT_ID'].isin(attrs['ACCOUNT_ID'])]['ACCOUNT_ID'].unique()
    if len(missing_customers) > 0:
        print(f"\n  {len(missing_customers):,} customers in transactions WITHOUT attributes:")

        missing_trans = trans[trans['ACCOUNT_ID'].isin(missing_customers)]
        missing_trans_count = len(missing_trans)
        missing_orders = missing_trans['ORDER_ID'].nunique()

        print(f"   - {missing_trans_count:,} transactions ({missing_trans_count/len(trans)*100:.2f}%)")
        print(f"   - {missing_orders:,} orders")

        customer_counts = [(acc_id, len(trans[trans['ACCOUNT_ID'] == acc_id]))
                          for acc_id in missing_customers]
        customer_counts_sorted = sorted(customer_counts, key=lambda x: x[1], reverse=True)

        top_customers = customer_counts_sorted[:10] if len(customer_counts_sorted) > 10 else customer_counts_sorted
        print(f"\n   Top ACCOUNT_IDs without attributes (ordered by transaction count):")
        for acc_id, n_trans in top_customers:
            print(f"     - {acc_id}: {n_trans} transactions")

        if len(missing_customers) > 10:
            print(f"     ... and {len(missing_customers) - 10} more")

    print("\nAttribute columns available:")
    for col, dtype in attrs.dtypes.items():
        n_unique = attrs[col].nunique()
        print(f"  {col:30s} {str(dtype):10s} {n_unique:6,} unique values")

    return trans, attrs, exact_dups_removed
