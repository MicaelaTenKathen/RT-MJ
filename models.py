"""
Baseline, NMF, EASE Models.
"""

import logging
import warnings

import pandas as pd
import numpy as np
from pathlib import Path
from scipy.sparse import csr_matrix
from sklearn.decomposition import NMF as SklearnNMF

warnings.filterwarnings("ignore", category=UserWarning, module="sklearn")

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

OUTPUT_DIR = Path('model_outputs')
OUTPUT_DIR.mkdir(exist_ok=True)

REQUIRED_TRANS_COLS = {'ACCOUNT_ID', 'SKU_ID', 'ORDER_ID', 'INVOICE_DATE', 'ITEMS_PHYS_CASES'}

np.random.seed(42)


def _validate_columns(df, required, name="DataFrame"):
    """Raise ValueError if required columns are missing."""
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing columns in {name}: {missing}")


def build_interaction_matrix(train):
    """
    Build sparse user-item interaction matrix.
    Value = log1p(frequency) + recency_boost for items bought in last 30 days.
    """
    logger.info("Building interaction matrix")

    max_date = train['INVOICE_DATE'].max()

    features = (train.groupby(['ACCOUNT_ID', 'SKU_ID'])
                .agg(
                    frequency=('ORDER_ID', 'nunique'),
                    last_purchase=('INVOICE_DATE', 'max'),
                    total_quantity=('ITEMS_PHYS_CASES', 'sum')
                )
                .reset_index())

    features['recency_days'] = (max_date - features['last_purchase']).dt.days
    features['interaction'] = np.log1p(features['frequency'])
    features.loc[features['recency_days'] < 30, 'interaction'] += 0.3

    unique_users = sorted(train['ACCOUNT_ID'].unique())
    unique_items = sorted(train['SKU_ID'].unique())

    user_to_idx = {u: i for i, u in enumerate(unique_users)}
    item_to_idx = {s: i for i, s in enumerate(unique_items)}
    idx_to_user = {i: u for u, i in user_to_idx.items()}
    idx_to_item = {i: s for s, i in item_to_idx.items()}

    row_indices = features['ACCOUNT_ID'].map(user_to_idx).values
    col_indices = features['SKU_ID'].map(item_to_idx).values
    data = features['interaction'].values

    n_users, n_items = len(unique_users), len(unique_items)
    X = csr_matrix((data, (row_indices, col_indices)), shape=(n_users, n_items))

    logger.info("  Shape: %s, nnz: %s, sparsity: %.2f%%",
                X.shape, f"{X.nnz:,}", (1 - X.nnz / (n_users * n_items)) * 100)

    mappings = {
        'user_to_idx': user_to_idx,
        'item_to_idx': item_to_idx,
        'idx_to_user': idx_to_user,
        'idx_to_item': idx_to_item,
    }

    return X, features, mappings


def build_k_lookup(attrs, default_k=5):
    """Pre-index dynamic K per customer (O(1) lookup instead of O(N) scan)."""
    if 'SkuDistintosPromediosXOrden' not in attrs.columns:
        return {}, default_k

    valid = attrs[['ACCOUNT_ID', 'SkuDistintosPromediosXOrden']].dropna(subset=['SkuDistintosPromediosXOrden'])
    k_values = np.clip(np.round(valid['SkuDistintosPromediosXOrden'].values * 1.5), 3, 12).astype(int)
    k_map = dict(zip(valid['ACCOUNT_ID'].values, k_values))

    return k_map, default_k


_k_cache = None


def get_dynamic_k(customer_id, attrs, default_k=5):
    """
    Get dynamic K for a customer.
    """
    global _k_cache
    if _k_cache is None or _k_cache['attrs_id'] != id(attrs):
        k_map, _ = build_k_lookup(attrs, default_k)
        _k_cache = {'attrs_id': id(attrs), 'k_map': k_map, 'default': default_k}

    return _k_cache['k_map'].get(customer_id, _k_cache['default'])


def build_fallback_popularity(train, attrs):
    """Build 2-tier fallback: global popularity + per-channel popularity."""
    logger.info("Building fallback popularity")

    global_pop = (train.groupby('SKU_ID')['ORDER_ID']
                  .nunique()
                  .sort_values(ascending=False)
                  .index.tolist())

    channel_pop = {}
    if 'canal' in attrs.columns:
        merged = train.merge(attrs[['ACCOUNT_ID', 'canal']], on='ACCOUNT_ID', how='left')
        for channel in merged['canal'].dropna().unique():
            channel_data = merged[merged['canal'] == channel]
            channel_pop[channel] = (channel_data.groupby('SKU_ID')['ORDER_ID']
                                    .nunique()
                                    .sort_values(ascending=False)
                                    .index.tolist())

    logger.info("  Global: %d SKUs, channels: %d", len(global_pop), len(channel_pop))

    return global_pop, channel_pop


def apply_fallback(customer_id, k, global_pop, channel_pop, attrs):
    """Cold-start fallback: try channel popularity first, then global."""
    recs = []
    scores = []
    customer_info = attrs[attrs['ACCOUNT_ID'] == customer_id]

    if len(customer_info) > 0 and 'canal' in customer_info.columns:
        channel = customer_info['canal'].values[0]
        if pd.notna(channel) and channel in channel_pop:
            channel_items = channel_pop[channel][:k]
            recs.extend(channel_items)
            scores.extend([0.1] * len(channel_items))

    if len(recs) < k:
        needed = k - len(recs)
        recs_set = set(recs)
        global_items = [sku for sku in global_pop if sku not in recs_set][:needed]
        recs.extend(global_items)
        scores.extend([0.05] * len(global_items))

    return recs[:k], scores[:k]


def fill_with_fallback(customer_id, recs, rec_scores, k, global_pop, channel_pop, attrs):
    """Fill incomplete recommendations up to k using 2-tier fallback."""
    needed = k - len(recs)
    if needed <= 0:
        return recs, rec_scores

    recs_set = set(recs)
    customer_info = attrs[attrs['ACCOUNT_ID'] == customer_id]

    if len(customer_info) > 0 and 'canal' in customer_info.columns:
        channel = customer_info['canal'].values[0]
        if pd.notna(channel) and channel in channel_pop:
            for sku in channel_pop[channel]:
                if sku not in recs_set and needed > 0:
                    recs.append(sku)
                    rec_scores.append(0.1)
                    recs_set.add(sku)
                    needed -= 1

    for sku in global_pop:
        if needed <= 0:
            break
        if sku not in recs_set:
            recs.append(sku)
            rec_scores.append(0.05)
            recs_set.add(sku)
            needed -= 1

    return recs, rec_scores


class BaselineFreqRecency:
    """score = freq + w_r * recency_score + w_q * log1p(qty)"""

    def __init__(self, recency_weight=0.3, quantity_weight=0.3):
        self.recency_weight = recency_weight
        self.quantity_weight = quantity_weight
        self.name = "Baseline Freq-Recency"

    def fit(self, train):
        max_date = train['INVOICE_DATE'].max()
        min_date = train['INVOICE_DATE'].min()
        max_days = (max_date - min_date).days

        self.features = (train.groupby(['ACCOUNT_ID', 'SKU_ID'])
                        .agg(
                            frequency=('ORDER_ID', 'nunique'),
                            total_quantity=('ITEMS_PHYS_CASES', 'sum'),
                            last_purchase=('INVOICE_DATE', 'max')
                        )
                        .reset_index())

        self.features['recency_days'] = (max_date - self.features['last_purchase']).dt.days
        self.features['recency_score'] = 1 - (self.features['recency_days'] / max(max_days, 1))

        self.features['score'] = (
            self.features['frequency'] +
            self.recency_weight * self.features['recency_score'] +
            self.quantity_weight * np.log1p(self.features['total_quantity'])
        )

        logger.info("  %s: fitted on %s (customer, SKU) pairs",
                     self.name, f"{len(self.features):,}")
        return self

    def recommend(self, customer_id, k, global_pop, channel_pop, attrs):
        customer_history = self.features[self.features['ACCOUNT_ID'] == customer_id]

        if len(customer_history) == 0:
            return apply_fallback(customer_id, k, global_pop, channel_pop, attrs)

        top_items = customer_history.nlargest(k, 'score')
        recs = top_items['SKU_ID'].tolist()
        scores = top_items['score'].tolist()

        if len(recs) < k:
            recs, scores = fill_with_fallback(customer_id, recs, scores, k, global_pop, channel_pop, attrs)

        return recs[:k], scores[:k]


class NMFModel:
    """NMF with blending: pred = reconstruction + blend_weight * raw_interactions"""

    def __init__(self, n_components=20, blend_weight=0.50):
        self.n_components = n_components
        self.blend_weight = blend_weight
        self.name = f"NMF (k={n_components}, blend={blend_weight})"

    def fit(self, X, mappings):
        logger.info("  %s: fitting...", self.name)

        self.mappings = mappings
        self.X = X

        nmf = SklearnNMF(
            n_components=self.n_components,
            init='nndsvda',
            max_iter=300,
            random_state=42,
        )

        self.W = nmf.fit_transform(X)
        self.H = nmf.components_
        self.X_reconstructed = self.W @ self.H

        self.X_blended = self.X_reconstructed + self.blend_weight * X.toarray()

        logger.info("    W: %s, H: %s, recon_err: %.2f",
                     self.W.shape, self.H.shape, nmf.reconstruction_err_)
        return self

    def recommend(self, customer_id, k, global_pop, channel_pop, attrs):
        if customer_id not in self.mappings['user_to_idx']:
            return apply_fallback(customer_id, k, global_pop, channel_pop, attrs)

        user_idx = self.mappings['user_to_idx'][customer_id]
        user_scores = self.X_blended[user_idx, :]

        top_indices = np.argsort(user_scores)[::-1][:k]
        recs = [self.mappings['idx_to_item'][i] for i in top_indices]
        rec_scores = [user_scores[i] for i in top_indices]

        if len(recs) < k:
            recs, rec_scores = fill_with_fallback(customer_id, recs, rec_scores, k, global_pop, channel_pop, attrs)

        return recs[:k], rec_scores[:k]


class EASEModel:
    """
    Closed-form item-item model.
    B_ij = -P_ij / P_jj  (i != j),  B_jj = 0
    where P = (X'X + lambda*I)^{-1}
    """

    def __init__(self, lambda_reg=50):
        self.lambda_reg = lambda_reg
        self.name = f"EASE (lambda={lambda_reg})"

    def fit(self, X, mappings):
        logger.info("  %s: fitting...", self.name)

        self.mappings = mappings
        self.X = X

        X_dense = X.toarray()

        G = X_dense.T @ X_dense
        G[np.diag_indices_from(G)] += self.lambda_reg

        P = np.linalg.inv(G)

        self.B = -P / np.diag(P)[np.newaxis, :]
        np.fill_diagonal(self.B, 0.0)

        logger.info("    B: %s, nnz: %s", self.B.shape, f"{np.count_nonzero(self.B):,}")
        return self

    def recommend(self, customer_id, k, global_pop, channel_pop, attrs):
        if customer_id not in self.mappings['user_to_idx']:
            return apply_fallback(customer_id, k, global_pop, channel_pop, attrs)

        user_idx = self.mappings['user_to_idx'][customer_id]
        user_vector = self.X[user_idx, :].toarray().flatten()
        user_scores = user_vector @ self.B

        top_indices = np.argsort(user_scores)[::-1][:k]
        recs = [self.mappings['idx_to_item'][i] for i in top_indices]
        rec_scores = [user_scores[i] for i in top_indices]

        if len(recs) < k:
            recs, rec_scores = fill_with_fallback(customer_id, recs, rec_scores, k, global_pop, channel_pop, attrs)

        return recs[:k], rec_scores[:k]
