"""Tests for synthetic data generation and real data loading."""

import numpy as np
import pandas as pd
import pytest

from src.data_gen import (
    CATEGORIES,
    DOCUMENTS,
    generate_services,
    generate_synthetic_dataset,
    load_config,
    load_real_data,
)


def test_data_generation_shapes_and_schemas():
    cfg = load_config()
    df_platforms, df_services, df_gt = generate_synthetic_dataset(cfg, save_raw=False)

    assert len(df_platforms) >= 40, "Should generate at least 40 platforms"
    assert len(df_services) >= 150, "Should generate at least 150 services"
    assert len(df_gt) == len(df_platforms), "Ground truth length must match platforms length"

    # Schema checks
    assert "platform_id" in df_platforms.columns
    assert "platform_name" in df_platforms.columns
    assert "service_description_text" in df_platforms.columns
    assert "is_synthetic" in df_platforms.columns
    assert bool(df_platforms["is_synthetic"].iloc[0]) is True

    for cat in CATEGORIES:
        assert f"coverage_{cat}" in df_platforms.columns

    for doc in DOCUMENTS:
        assert doc in df_platforms.columns


def test_seed_reproducibility():
    cfg = load_config()
    p1, _, _ = generate_synthetic_dataset(cfg, save_raw=False)
    p2, _, _ = generate_synthetic_dataset(cfg, save_raw=False)

    # Identical shapes and platform IDs
    pd.testing.assert_frame_equal(p1.drop(columns=["service_description_text"]), p2.drop(columns=["service_description_text"]))


def test_ground_truth_isolation():
    cfg = load_config()
    df_platforms, _, df_gt = generate_synthetic_dataset(cfg, save_raw=False)

    # The ground truth cluster label must NOT be present in platforms dataframe
    assert "ground_truth_cluster" not in df_platforms.columns
    assert "cluster_name" not in df_platforms.columns
    assert "ground_truth_cluster" in df_gt.columns
