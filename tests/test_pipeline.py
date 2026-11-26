"""Tests for end-to-end pipeline execution."""

from pathlib import Path
import pytest

from src.pipeline import run_pipeline


def test_pipeline_end_to_end(tmp_path):
    results = run_pipeline(skip_gen=True)

    assert "df_platforms" in results
    assert "X_pca" in results
    assert "cluster_df" in results
    assert "benchmark_df" in results
    assert "recommendations_df" in results

    assert len(results["df_platforms"]) > 0
    assert results["X_pca"].shape[0] == len(results["df_platforms"])
    assert not results["benchmark_df"].empty
    assert not results["recommendations_df"].empty
