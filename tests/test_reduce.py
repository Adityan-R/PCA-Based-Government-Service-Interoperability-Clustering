"""Tests for PCA reduction, explained variance threshold, and loadings."""

import numpy as np
import pytest

from src.data_gen import generate_synthetic_dataset, load_config
from src.features import PlatformFeaturePipeline
from src.reduce import PCAReducer, reduce_umap


def test_pca_variance_threshold_and_loadings():
    cfg = load_config()
    df_platforms, _, _ = generate_synthetic_dataset(cfg, save_raw=False)
    pipeline = PlatformFeaturePipeline(cfg)
    X, feat_names, _, _ = pipeline.fit_transform(df_platforms)

    reducer = PCAReducer(cfg)
    target_var = float(cfg["reduction"]["pca_target_variance"])
    X_pca, scree_df, loadings_df = reducer.fit_transform(X, feat_names)

    # Dimensionality reduction check
    assert X_pca.shape[1] < X.shape[1], "PCA must reduce dimensionality"
    assert X_pca.shape[0] == X.shape[0]

    # Target variance check
    cum_var_selected = scree_df[scree_df["is_selected"]]["explained_variance_ratio"].sum()
    assert cum_var_selected >= target_var - 0.05, f"Cumulative variance {cum_var_selected} must meet target {target_var}"

    # Loadings structure check
    assert not loadings_df.empty
    assert "component" in loadings_df.columns
    assert "feature_name" in loadings_df.columns
    assert "loading_weight" in loadings_df.columns


def test_umap_reduction():
    rng = np.random.default_rng(42)
    X_dummy = rng.normal(size=(20, 30))
    X_u = reduce_umap(X_dummy, n_components=2, random_state=42)

    assert X_u.shape == (20, 2), "UMAP output must match (n_samples, 2)"
    assert not np.isnan(X_u).any()
