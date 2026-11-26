"""Tests for feature preprocessing, missing value handling, and block fusion."""

import numpy as np
import pandas as pd
import pytest

from src.data_gen import generate_synthetic_dataset, load_config
from src.features import PlatformFeaturePipeline


def test_feature_pipeline_missing_imputation_and_dims():
    cfg = load_config()
    df_platforms, _, _ = generate_synthetic_dataset(cfg, save_raw=False)

    pipeline = PlatformFeaturePipeline(cfg)
    X, feat_names, pids, pnames = pipeline.fit_transform(df_platforms)

    # Dimensionality check: must be in 100-300 range
    assert 100 <= X.shape[1] <= 300, f"Raw feature dimension {X.shape[1]} should be between 100 and 300"
    assert X.shape[0] == len(df_platforms)

    # Imputation check: no NaNs or Infs
    assert not np.isnan(X).any(), "Feature matrix must not contain any NaNs after imputation"
    assert not np.isinf(X).any(), "Feature matrix must not contain any Infs"
    assert len(feat_names) == X.shape[1]
    assert len(pids) == len(df_platforms)


def test_feature_pipeline_transform_consistency():
    cfg = load_config()
    df_platforms, _, _ = generate_synthetic_dataset(cfg, save_raw=False)

    pipeline = PlatformFeaturePipeline(cfg)
    X_fit, names_fit, _, _ = pipeline.fit_transform(df_platforms)
    X_trans, _, _ = pipeline.transform(df_platforms)

    np.testing.assert_allclose(X_fit, X_trans, rtol=1e-5)
