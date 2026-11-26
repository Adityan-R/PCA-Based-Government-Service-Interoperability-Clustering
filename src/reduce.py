"""Dimensionality reduction module.

Implements PCA (variance-based selection, scree analysis, top loadings explainability)
alongside UMAP and raw-feature baselines for comparison and justification.
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
from sklearn.decomposition import PCA
import umap

from src.data_gen import load_config


class PCAReducer:
    """PCA Dimensionality Reduction with variance thresholding and explainable loadings."""

    def __init__(self, config: dict[str, Any] | None = None) -> None:
        if config is None:
            config = load_config()
        self.config = config
        red_cfg = config.get("reduction", {})
        self.target_variance = float(red_cfg.get("pca_target_variance", 0.92))
        self.seed = int(config.get("seed", 42))
        self.top_loadings_k = int(red_cfg.get("top_loadings_per_component", 5))

        self.pca_full: PCA | None = None
        self.pca_selected: PCA | None = None
        self.n_components_: int = 0
        self.explained_variance_ratio_: np.ndarray | None = None
        self.cumulative_variance_ratio_: np.ndarray | None = None
        self.feature_names_: List[str] = []

    def fit_transform(
        self, X: np.ndarray, feature_names: List[str]
    ) -> Tuple[np.ndarray, pd.DataFrame, pd.DataFrame]:
        """Fit PCA and select components achieving cumulative explained variance.

        Returns:
            X_pca: Reduced matrix (n_samples, n_selected_components)
            scree_df: Component variance and cumulative variance breakdown
            loadings_df: Top feature loadings per principal component
        """
        self.feature_names_ = feature_names
        max_comps = min(X.shape[0] - 1, X.shape[1])

        # Step 1: Fit full PCA to compute the full spectrum
        self.pca_full = PCA(n_components=max_comps, random_state=self.seed)
        self.pca_full.fit(X)

        cum_var = np.cumsum(self.pca_full.explained_variance_ratio_)
        # Find index where target variance is achieved
        comp_idx = np.where(cum_var >= self.target_variance)[0]
        if len(comp_idx) > 0:
            self.n_components_ = int(comp_idx[0] + 1)
        else:
            self.n_components_ = max_comps

        # Ensure at least 2 components for 2D visualization
        self.n_components_ = max(2, self.n_components_)

        # Step 2: Fit selected PCA model
        self.pca_selected = PCA(n_components=self.n_components_, random_state=self.seed)
        X_pca = self.pca_selected.fit_transform(X)

        self.explained_variance_ratio_ = self.pca_selected.explained_variance_ratio_
        self.cumulative_variance_ratio_ = np.cumsum(self.explained_variance_ratio_)

        # Step 3: Build Scree DataFrame
        scree_records = []
        for i in range(len(self.pca_full.explained_variance_ratio_)):
            scree_records.append(
                {
                    "component": i + 1,
                    "component_label": f"PC{i+1}",
                    "explained_variance_ratio": float(np.round(self.pca_full.explained_variance_ratio_[i], 5)),
                    "cumulative_variance_ratio": float(np.round(cum_var[i], 5)),
                    "is_selected": bool(i < self.n_components_),
                }
            )
        scree_df = pd.DataFrame(scree_records)

        # Step 4: Extract top feature loadings per component
        loadings_records = []
        components_mat = self.pca_selected.components_  # shape: (n_components, n_features)

        for comp_i in range(self.n_components_):
            comp_loadings = components_mat[comp_i]
            # Top positive and negative contributors by magnitude
            sorted_indices = np.argsort(np.abs(comp_loadings))[::-1]
            for rank, feat_idx in enumerate(sorted_indices[: self.top_loadings_k], 1):
                loadings_records.append(
                    {
                        "component": comp_i + 1,
                        "component_label": f"PC{comp_i+1}",
                        "rank": rank,
                        "feature_name": feature_names[feat_idx],
                        "loading_weight": float(np.round(comp_loadings[feat_idx], 4)),
                        "abs_loading": float(np.round(np.abs(comp_loadings[feat_idx]), 4)),
                    }
                )

        loadings_df = pd.DataFrame(loadings_records)
        return X_pca, scree_df, loadings_df


def reduce_umap(
    X: np.ndarray,
    n_components: int = 2,
    n_neighbors: int = 12,
    min_dist: float = 0.1,
    random_state: int = 42,
) -> np.ndarray:
    """Non-linear dimensionality reduction using UMAP."""
    reducer = umap.UMAP(
        n_components=n_components,
        n_neighbors=n_neighbors,
        min_dist=min_dist,
        random_state=random_state,
        compatibility_layout=True,
    )
    return reducer.fit_transform(X)


def run_reduction_pipeline(
    X: np.ndarray,
    feature_names: List[str],
    config: dict[str, Any] | None = None,
) -> Dict[str, Any]:
    """Execute PCA, UMAP, and save artifacts."""
    if config is None:
        config = load_config()

    seed = int(config.get("seed", 42))
    pca_reducer = PCAReducer(config)
    X_pca, scree_df, loadings_df = pca_reducer.fit_transform(X, feature_names)

    # Compute 2D UMAP projection
    u_cfg = config.get("reduction", {})
    X_umap = reduce_umap(
        X,
        n_components=int(u_cfg.get("umap_n_components", 2)),
        n_neighbors=int(u_cfg.get("umap_n_neighbors", 12)),
        min_dist=float(u_cfg.get("umap_min_dist", 0.1)),
        random_state=seed,
    )

    # Save artifacts
    proc_dir = Path(config["paths"]["processed_dir"])
    proc_dir.mkdir(parents=True, exist_ok=True)

    np.save(config["paths"]["pca_features_npy"], X_pca)
    scree_df.to_csv(config["paths"]["scree_plot_csv"], index=False)
    loadings_df.to_csv(config["paths"]["pca_loadings_csv"], index=False)

    return {
        "X_pca": X_pca,
        "X_umap": X_umap,
        "n_components": pca_reducer.n_components_,
        "scree_df": scree_df,
        "loadings_df": loadings_df,
        "target_variance": pca_reducer.target_variance,
    }


if __name__ == "__main__":
    cfg = load_config()
    feat_df = pd.read_csv(cfg["paths"]["features_csv"])
    feat_names = [c for c in feat_df.columns if c not in ["platform_id", "platform_name"]]
    X_raw = feat_df[feat_names].to_numpy(dtype=float)

    results = run_reduction_pipeline(X_raw, feat_names, cfg)
    print(f"PCA reduced from {X_raw.shape[1]} to {results['n_components']} components.")
    print(f"PCA shape: {results['X_pca'].shape}, UMAP shape: {results['X_umap'].shape}")
    print("\nTop 5 loadings for PC1:")
    print(results["loadings_df"][results["loadings_df"]["component"] == 1])
