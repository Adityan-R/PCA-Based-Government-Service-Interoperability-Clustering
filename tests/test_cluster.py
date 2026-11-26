"""Tests for clustering algorithms, labels validity, and intrinsic metrics."""

import numpy as np
import pytest

from src.cluster import ClusterSuite, compute_intrinsic_metrics, find_optimal_kmeans_k
from src.data_gen import load_config


def test_kmeans_and_optimal_k():
    cfg = load_config()
    suite = ClusterSuite(cfg)

    rng = np.random.default_rng(42)
    # Generate 4 well-separated synthetic blobs
    c1 = rng.normal(loc=-5.0, scale=0.5, size=(10, 10))
    c2 = rng.normal(loc=0.0, scale=0.5, size=(10, 10))
    c3 = rng.normal(loc=5.0, scale=0.5, size=(10, 10))
    c4 = rng.normal(loc=10.0, scale=0.5, size=(10, 10))
    X = np.vstack([c1, c2, c3, c4])

    best_k, k_df = find_optimal_kmeans_k(X, k_min=3, k_max=6, seed=42)
    assert best_k == 4, f"Optimal k should be 4 for 4 separated blobs, got {best_k}"

    labels, metrics = suite.run_kmeans(X, k=4, seed=42)
    assert len(labels) == len(X)
    assert metrics["silhouette"] > 0.5, "Silhouette score for well-separated clusters should be high"
    assert metrics["davies_bouldin"] < 1.0


def test_agglomerative_and_hdbscan():
    cfg = load_config()
    suite = ClusterSuite(cfg)

    rng = np.random.default_rng(42)
    X = rng.normal(size=(30, 8))

    agg_labels, agg_metrics = suite.run_agglomerative(X, k=5)
    assert len(set(agg_labels)) == 5
    assert len(agg_labels) == 30

    hdb_labels, hdb_metrics = suite.run_hdbscan(X)
    assert len(hdb_labels) == 30
    assert "silhouette" in hdb_metrics
    assert "noise_ratio" in hdb_metrics
