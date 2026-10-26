"""Low-rank user/item taste model."""

from __future__ import annotations

from dataclasses import dataclass
import numpy as np
from scipy.sparse import coo_matrix
from scipy.sparse.linalg import svds

from .cache import CachedEvents


@dataclass
class SVDModel:
    user_vectors: np.ndarray
    item_vectors: np.ndarray
    item_degree: np.ndarray
    user_degree: np.ndarray

    @property
    def rank(self) -> int:
        return int(self.item_vectors.shape[1])

    def user_vector(self, user: int, history_items: np.ndarray | None = None) -> np.ndarray:
        if 0 <= user < len(self.user_vectors):
            return self.user_vectors[user]
        if history_items is not None:
            valid = history_items[history_items >= 0]
            valid = valid[valid < len(self.item_vectors)]
            if len(valid):
                return self.item_vectors[valid].mean(axis=0)
        return np.zeros(self.rank, dtype=np.float32)


def fit_svd(click_log: np.ndarray, n_users: int, n_items: int, rank: int = 11, binary: bool = True) -> SVDModel:
    if len(click_log) == 0:
        raise ValueError("cannot fit SVD without clicks")
    pairs = click_log[:, :2].astype(np.int64, copy=False)
    if binary:
        pairs = np.unique(pairs, axis=0)
    values = np.ones(len(pairs), dtype=np.float64)
    rows, cols = pairs[:, 0], pairs[:, 1]
    user_degree = np.bincount(rows, minlength=n_users).astype(np.float64) + 1.0
    item_degree = np.bincount(cols, minlength=n_items).astype(np.float64) + 1.0
    values = values / np.sqrt(user_degree[rows] * item_degree[cols])
    matrix = coo_matrix((values, (rows, cols)), shape=(n_users, n_items)).tocsr()
    max_rank = min(matrix.shape) - 1
    actual_rank = max(1, min(int(rank), max_rank))
    u, singular, vt = svds(matrix, k=actual_rank, which="LM", return_singular_vectors=True, random_state=20261024)
    order = np.argsort(singular)[::-1]
    u = u[:, order]
    singular = singular[order]
    vt = vt[order]
    user_vectors = (u * singular[None, :]).astype(np.float32)
    item_vectors = vt.T.astype(np.float32)
    return SVDModel(user_vectors, item_vectors, item_degree.astype(np.float32), user_degree.astype(np.float32))


def fit_impression_svd(events: CachedEvents, boundary: int, n_users: int, n_items: int, rank: int = 10) -> SVDModel:
    """Fit the weaker user-item impression matrix using only pre-B events."""
    mask = events.timestamp < int(boundary)
    event_ids = np.flatnonzero(mask)
    rows = np.repeat(events.user_id[event_ids], events.candidates.shape[1]).astype(np.int64)
    cols = events.candidates[event_ids].reshape(-1).astype(np.int64)
    pairs = np.unique(np.column_stack([rows, cols]), axis=0)
    values = np.ones(len(pairs), dtype=np.float64)
    rows, cols = pairs[:, 0], pairs[:, 1]
    user_degree = np.bincount(rows, minlength=n_users).astype(np.float64) + 1.0
    item_degree = np.bincount(cols, minlength=n_items).astype(np.float64) + 1.0
    values /= np.sqrt(user_degree[rows] * item_degree[cols])
    matrix = coo_matrix((values, (rows, cols)), shape=(n_users, n_items)).tocsr()
    actual_rank = max(1, min(int(rank), min(matrix.shape) - 1))
    u, singular, vt = svds(matrix, k=actual_rank, which="LM", return_singular_vectors=True, random_state=20261024)
    order = np.argsort(singular)[::-1]
    u, singular, vt = u[:, order], singular[order], vt[order]
    return SVDModel((u * singular[None, :]).astype(np.float32), vt.T.astype(np.float32), item_degree.astype(np.float32), user_degree.astype(np.float32))
