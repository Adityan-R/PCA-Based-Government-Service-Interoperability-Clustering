"""Tests for pairwise overlap matrix range [0, 1] and recommendation generation."""

import numpy as np
import pandas as pd
import pytest

from src.data_gen import generate_synthetic_dataset, load_config
from src.overlap import compute_pairwise_overlap_matrix, generate_recommendations


def test_overlap_matrix_bounds_and_symmetry():
    cfg = load_config()
    df_platforms, _, _ = generate_synthetic_dataset(cfg, save_raw=False)

    mat, pids, pnames = compute_pairwise_overlap_matrix(df_platforms)

    # Dimensionality check
    n = len(df_platforms)
    assert mat.shape == (n, n), f"Matrix shape {mat.shape} must be ({n}, {n})"

    # Range [0, 1] check
    assert np.all(mat >= 0.0), "Overlap scores must be non-negative"
    assert np.all(mat <= 1.0), "Overlap scores must not exceed 1.0"

    # Diagonal must be 1.0
    np.testing.assert_allclose(np.diag(mat), np.ones(n))

    # Symmetry check
    np.testing.assert_allclose(mat, mat.T, atol=1e-6)


def test_recommendations_generation():
    cfg = load_config()
    df_platforms, _, _ = generate_synthetic_dataset(cfg, save_raw=False)
    mat, _, _ = compute_pairwise_overlap_matrix(df_platforms)

    # Dummy cluster assignments
    cluster_labels = np.array([i % 5 for i in range(len(df_platforms))])
    recs = generate_recommendations(df_platforms, cluster_labels, mat, cfg)

    assert not recs.empty, "Recommendations table should not be empty"
    assert "platform_a" in recs.columns
    assert "platform_b" in recs.columns
    assert "overlap_score" in recs.columns
    assert "suggested_action" in recs.columns

    # Overlap scores must be within [0, 1]
    assert (recs["overlap_score"] >= 0.0).all()
    assert (recs["overlap_score"] <= 1.0).all()
    assert (recs["overlap_pct"] >= 0.0).all()
    assert (recs["overlap_pct"] <= 100.0).all()
