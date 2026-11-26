"""End-to-End Orchestrator Pipeline for SIH26129.

Executes all 5 stages: Data -> Features -> PCA/UMAP -> Clustering & Benchmarking -> Overlap & Recommendations.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path
from typing import Any, Dict

# Ensure project root is on sys.path
root_dir = str(Path(__file__).resolve().parent.parent)
if root_dir not in sys.path:
    sys.path.insert(0, root_dir)

import numpy as np
import pandas as pd

from src.cluster import ClusterSuite
from src.data_gen import (
    generate_synthetic_dataset,
    load_config,
    load_real_data,
)
from src.evaluate import run_ablation_benchmarking
from src.features import process_and_save_features
from src.overlap import run_overlap_pipeline
from src.reduce import run_reduction_pipeline


def run_pipeline(
    config_path: str = "config.yaml",
    real_data_path: str | None = None,
    skip_gen: bool = False,
) -> dict[str, Any]:
    """Execute the full end-to-end interoperability discovery pipeline."""
    start_time = time.time()
    config = load_config(config_path)

    print("=" * 70)
    print("🚀 SIH26129: PCA-BASED INTEROPERABILITY CLUSTERING PIPELINE")
    print("=" * 70)

    # Stage 1: Data Acquisition
    print("\n[Stage 1/5] Ingesting Data...")
    if real_data_path:
        print(f"Loading REAL platform data from: {real_data_path}")
        df_platforms = load_real_data(real_data_path)
        df_ground_truth = pd.DataFrame(
            {
                "platform_id": df_platforms["platform_id"],
                "platform_name": df_platforms["platform_name"],
                "ground_truth_cluster": -1,
            }
        )
    elif skip_gen and Path(config["paths"]["platforms_csv"]).exists():
        print(f"Loading existing platforms from {config['paths']['platforms_csv']}...")
        df_platforms = pd.read_csv(config["paths"]["platforms_csv"])
        df_ground_truth = pd.read_csv(config["paths"]["ground_truth_csv"])
    else:
        print("Generating realistic synthetic government platforms and services...")
        df_platforms, df_services, df_ground_truth = generate_synthetic_dataset(config, save_raw=True)

    print(f"✓ Platforms loaded: {len(df_platforms)} (is_synthetic={df_platforms['is_synthetic'].iloc[0]})")

    # Stage 2: Feature Engineering
    print("\n[Stage 2/5] Preprocessing & Constructing Multi-Block Feature Matrix...")
    X_raw, feature_names, feat_df = process_and_save_features(df_platforms, config)
    print(f"✓ Feature matrix constructed: {X_raw.shape[0]} platforms x {X_raw.shape[1]} raw dimensions")

    # Stage 3: Dimensionality Reduction (PCA & UMAP)
    print("\n[Stage 3/5] Applying PCA & Generating Baseline Projections...")
    reduction_artifacts = run_reduction_pipeline(X_raw, feature_names, config)
    X_pca = reduction_artifacts["X_pca"]
    X_umap = reduction_artifacts["X_umap"]
    n_pca_comps = reduction_artifacts["n_components"]
    print(f"✓ PCA reduced features from {X_raw.shape[1]} to {n_pca_comps} components (Retained: {reduction_artifacts['target_variance']*100:.1f}% variance)")

    # Stage 4: Clustering & Benchmarking
    print("\n[Stage 4/5] Executing Clustering Algorithms & Ground Truth Benchmarking...")
    suite = ClusterSuite(config)
    default_k = config.get("clustering", {}).get("default_k", 7)

    km_labels, km_metrics = suite.run_kmeans(X_pca, k=default_k)
    agg_labels, agg_metrics = suite.run_agglomerative(X_pca, k=default_k)
    hdb_labels, hdb_metrics = suite.run_hdbscan(X_pca)

    # Save cluster labels to disk
    cluster_df = pd.DataFrame(
        {
            "platform_id": df_platforms["platform_id"],
            "platform_name": df_platforms["platform_name"],
            "tier": df_platforms["tier"] if "tier" in df_platforms.columns else "State",
            "cluster_kmeans": km_labels,
            "cluster_agglomerative": agg_labels,
            "cluster_hdbscan": hdb_labels,
            "pca_1": X_pca[:, 0],
            "pca_2": X_pca[:, 1],
            "umap_1": X_umap[:, 0],
            "umap_2": X_umap[:, 1],
        }
    )
    cluster_df.to_csv(config["paths"]["cluster_results_csv"], index=False)

    # Run comprehensive ablation benchmark table
    bench_df = run_ablation_benchmarking(X_raw, df_ground_truth, config)
    print("✓ Benchmark matrix computed across Raw, PCA, and UMAP.")

    # Stage 5: Overlap Matrix & Policy Recommendations
    print("\n[Stage 5/5] Calculating Pairwise Overlap & Generating Policy Actions...")
    overlap_mat, rec_df = run_overlap_pipeline(df_platforms, km_labels, config)
    print(f"✓ Generated {len(rec_df)} actionable interoperability recommendations.")

    elapsed = time.time() - start_time
    print("\n" + "=" * 70)
    print(f"PIPELINE COMPLETED SUCCESSFULLY IN {elapsed:.2f} SECONDS")
    print(f"Artifacts saved to: {config['paths']['processed_dir']}/")
    print("=" * 70)

    return {
        "df_platforms": df_platforms,
        "X_raw": X_raw,
        "X_pca": X_pca,
        "X_umap": X_umap,
        "cluster_df": cluster_df,
        "benchmark_df": bench_df,
        "recommendations_df": rec_df,
        "overlap_matrix": overlap_mat,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="SIH26129 Interoperability Clustering Pipeline")
    parser.add_argument("--config", default="config.yaml", help="Path to config yaml")
    parser.add_argument("--real-data", default=None, help="Path to real platform features CSV")
    parser.add_argument("--skip-gen", action="store_true", help="Skip synthetic data generation if already exists")
    args = parser.parse_args()

    run_pipeline(
        config_path=args.config,
        real_data_path=args.real_data,
        skip_gen=args.skip_gen,
    )


if __name__ == "__main__":
    main()
