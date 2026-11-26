"""Clustering algorithms and intrinsic validation metrics.

Implements K-Means (with automated k selection via silhouette & elbow),
Agglomerative Hierarchical Clustering, and HDBSCAN density clustering.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple

# Ensure project root is on sys.path
root_dir = str(Path(__file__).resolve().parent.parent)
if root_dir not in sys.path:
    sys.path.insert(0, root_dir)

import numpy as np
import pandas as pd
from sklearn.cluster import AgglomerativeClustering, HDBSCAN, KMeans
from sklearn.metrics import (
    calinski_harabasz_score,
    davies_bouldin_score,
    silhouette_score,
)

from src.data_gen import load_config


def compute_intrinsic_metrics(X: np.ndarray, labels: np.ndarray) -> Dict[str, float]:
    """Calculate Silhouette, Davies-Bouldin, and Calinski-Harabasz metrics.

    Handles noise (-1 labels in HDBSCAN) gracefully.
    """
    # Remove noise label (-1) for cluster quality evaluation if present
    non_noise_mask = labels != -1
    valid_labels = labels[non_noise_mask]
    valid_X = X[non_noise_mask]

    n_clusters = len(set(valid_labels))
    if n_clusters < 2 or len(valid_labels) <= n_clusters:
        return {
            "silhouette": -1.0,
            "davies_bouldin": 99.0,
            "calinski_harabasz": 0.0,
            "n_clusters": float(n_clusters),
            "noise_ratio": float(np.mean(labels == -1)),
        }

    sil = float(np.round(silhouette_score(valid_X, valid_labels), 4))
    db = float(np.round(davies_bouldin_score(valid_X, valid_labels), 4))
    ch = float(np.round(calinski_harabasz_score(valid_X, valid_labels), 2))
    noise_ratio = float(np.round(np.mean(labels == -1), 4))

    return {
        "silhouette": sil,
        "davies_bouldin": db,
        "calinski_harabasz": ch,
        "n_clusters": float(n_clusters),
        "noise_ratio": noise_ratio,
    }


def find_optimal_kmeans_k(
    X: np.ndarray,
    k_min: int = 3,
    k_max: int = 12,
    seed: int = 42,
) -> Tuple[int, pd.DataFrame]:
    """Scan k range using Silhouette and Inertia (Elbow) to find optimal cluster count."""
    k_records = []
    best_k = k_min
    best_sil = -2.0

    for k in range(k_min, min(k_max + 1, len(X))):
        km = KMeans(n_clusters=k, random_state=seed, n_init=10)
        labels = km.fit_predict(X)
        sil = float(silhouette_score(X, labels))
        db = float(davies_bouldin_score(X, labels))
        ch = float(calinski_harabasz_score(X, labels))
        inertia = float(km.inertia_)

        k_records.append(
            {
                "k": k,
                "silhouette": float(np.round(sil, 4)),
                "davies_bouldin": float(np.round(db, 4)),
                "calinski_harabasz": float(np.round(ch, 2)),
                "inertia": float(np.round(inertia, 2)),
            }
        )

        if sil > best_sil:
            best_sil = sil
            best_k = k

    return best_k, pd.DataFrame(k_records)


class ClusterSuite:
    """Manages training and evaluation across KMeans, Agglomerative, and HDBSCAN."""

    def __init__(self, config: dict[str, Any] | None = None) -> None:
        if config is None:
            config = load_config()
        self.config = config
        c_cfg = config.get("clustering", {})
        self.k_min = int(c_cfg.get("k_min", 3))
        self.k_max = int(c_cfg.get("k_max", 12))
        self.default_k = int(c_cfg.get("default_k", 7))
        self.hdb_min_cluster_size = int(c_cfg.get("hdbscan_min_cluster_size", 3))
        self.hdb_min_samples = int(c_cfg.get("hdbscan_min_samples", 2))
        self.linkage = c_cfg.get("agglomerative_linkage", "ward")
        self.seed = int(config.get("seed", 42))

    def run_kmeans(self, X: np.ndarray, k: int | None = None, seed: int | None = None) -> Tuple[np.ndarray, Dict[str, float]]:
        """Fit K-Means clustering."""
        s = self.seed if seed is None else seed
        if k is None:
            k, _ = find_optimal_kmeans_k(X, self.k_min, self.k_max, seed=s)
        km = KMeans(n_clusters=k, random_state=s, n_init=10)
        labels = km.fit_predict(X)
        metrics = compute_intrinsic_metrics(X, labels)
        metrics["k_chosen"] = float(k)
        return labels, metrics

    def run_agglomerative(self, X: np.ndarray, k: int | None = None) -> Tuple[np.ndarray, Dict[str, float]]:
        """Fit Agglomerative Hierarchical clustering."""
        target_k = self.default_k if k is None else k
        agg = AgglomerativeClustering(n_clusters=target_k, linkage=self.linkage)
        labels = agg.fit_predict(X)
        metrics = compute_intrinsic_metrics(X, labels)
        metrics["k_chosen"] = float(target_k)
        return labels, metrics

    def run_hdbscan(self, X: np.ndarray) -> Tuple[np.ndarray, Dict[str, float]]:
        """Fit HDBSCAN density-based clustering."""
        hdb = HDBSCAN(
            min_cluster_size=self.hdb_min_cluster_size,
            min_samples=self.hdb_min_samples,
            copy=True,
        )
        labels = hdb.fit_predict(X)
        metrics = compute_intrinsic_metrics(X, labels)
        return labels, metrics


if __name__ == "__main__":
    cfg = load_config()
    X_pca = np.load(cfg["paths"]["pca_features_npy"])
    suite = ClusterSuite(cfg)

    k_opt, k_df = find_optimal_kmeans_k(X_pca, k_min=3, k_max=12, seed=cfg["seed"])
    print(f"Optimal KMeans k: {k_opt}")
    print(k_df)

    km_labels, km_met = suite.run_kmeans(X_pca, k=k_opt)
    agg_labels, agg_met = suite.run_agglomerative(X_pca, k=k_opt)
    hdb_labels, hdb_met = suite.run_hdbscan(X_pca)

    print(f"\nKMeans Metrics: {km_met}")
    print(f"Agglomerative Metrics: {agg_met}")
    print(f"HDBSCAN Metrics: {hdb_met}")
