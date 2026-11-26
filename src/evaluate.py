"""Ground truth validation, stability testing, and ablation benchmarking.

SIH26129: Evaluates clustering against hidden ground truth via ARI & NMI,
evaluates seed stability, and benchmarks Raw vs PCA (various dims) vs UMAP.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, Dict, List

# Ensure project root is on sys.path
root_dir = str(Path(__file__).resolve().parent.parent)
if root_dir not in sys.path:
    sys.path.insert(0, root_dir)

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score

from src.cluster import ClusterSuite
from src.data_gen import load_config
from src.reduce import reduce_umap


def compute_external_metrics(
    true_labels: np.ndarray,
    predicted_labels: np.ndarray,
) -> Dict[str, float]:
    """Calculate Adjusted Rand Index (ARI) and Normalized Mutual Information (NMI)."""
    ari = float(np.round(adjusted_rand_score(true_labels, predicted_labels), 4))
    nmi = float(np.round(normalized_mutual_info_score(true_labels, predicted_labels), 4))
    return {"adjusted_rand_index": ari, "normalized_mutual_info": nmi}


def test_seed_stability(
    X: np.ndarray,
    k: int,
    seeds: List[int],
) -> Dict[str, float]:
    """Test clustering stability across different random seeds using pairwise ARI."""
    suite = ClusterSuite()
    all_labels = []
    for s in seeds:
        lbls, _ = suite.run_kmeans(X, k=k, seed=s)
        all_labels.append(lbls)

    pairwise_aris = []
    for i in range(len(seeds)):
        for j in range(i + 1, len(seeds)):
            pairwise_aris.append(adjusted_rand_score(all_labels[i], all_labels[j]))

    mean_stability = float(np.round(np.mean(pairwise_aris), 4)) if pairwise_aris else 1.0
    std_stability = float(np.round(np.std(pairwise_aris), 4)) if pairwise_aris else 0.0

    return {
        "mean_seed_stability_ari": mean_stability,
        "std_seed_stability_ari": std_stability,
    }


def run_ablation_benchmarking(
    X_raw: np.ndarray,
    ground_truth_df: pd.DataFrame,
    config: dict[str, Any] | None = None,
) -> pd.DataFrame:
    """Run full ablation study across representations and algorithms."""
    if config is None:
        config = load_config()

    seed = int(config.get("seed", 42))
    stability_seeds = config.get("clustering", {}).get("stability_seeds", [42, 101, 202, 303, 404])
    true_labels = ground_truth_df["ground_truth_cluster"].to_numpy()

    suite = ClusterSuite(config)

    # 1. Prepare representations
    representations: Dict[str, Tuple[str, int, np.ndarray]] = {}

    # Raw features
    representations["Raw (Unreduced)"] = ("Raw", X_raw.shape[1], X_raw)

    # PCA at various components
    pca_comp_counts = [5, 10, 15, 20, 25]
    for n_comp in pca_comp_counts:
        pca_model = PCA(n_components=n_comp, random_state=seed)
        X_p = pca_model.fit_transform(X_raw)
        representations[f"PCA ({n_comp} dims)"] = ("PCA", n_comp, X_p)

    # PCA at target variance (e.g. 92%)
    target_var = float(config.get("reduction", {}).get("pca_target_variance", 0.92))
    pca_full = PCA(random_state=seed).fit(X_raw)
    cum_var = np.cumsum(pca_full.explained_variance_ratio_)
    opt_n = int(np.where(cum_var >= target_var)[0][0] + 1)
    X_pca_opt = PCA(n_components=opt_n, random_state=seed).fit_transform(X_raw)
    representations[f"PCA (92% Var, {opt_n} dims)"] = ("PCA", opt_n, X_pca_opt)

    # UMAP (2 dims)
    X_u = reduce_umap(X_raw, n_components=2, random_state=seed)
    representations["UMAP (2 dims)"] = ("UMAP", 2, X_u)

    # 2. Benchmark evaluation loop
    benchmark_rows = []

    for repr_name, (repr_type, dims, X_repr) in representations.items():
        # Evaluate algorithms: KMeans, Agglomerative, HDBSCAN
        for algo in ["K-Means", "Agglomerative", "HDBSCAN"]:
            if algo == "K-Means":
                lbls, metrics = suite.run_kmeans(X_repr, k=suite.default_k, seed=seed)
                stab = test_seed_stability(X_repr, k=suite.default_k, seeds=stability_seeds)
            elif algo == "Agglomerative":
                lbls, metrics = suite.run_agglomerative(X_repr, k=suite.default_k)
                stab = {"mean_seed_stability_ari": 1.0, "std_seed_stability_ari": 0.0}
            else:  # HDBSCAN
                lbls, metrics = suite.run_hdbscan(X_repr)
                stab = {"mean_seed_stability_ari": 1.0, "std_seed_stability_ari": 0.0}

            ext_metrics = compute_external_metrics(true_labels, lbls)

            benchmark_rows.append(
                {
                    "representation": repr_name,
                    "representation_type": repr_type,
                    "dimensions": dims,
                    "algorithm": algo,
                    "silhouette": metrics["silhouette"],
                    "davies_bouldin": metrics["davies_bouldin"],
                    "calinski_harabasz": metrics["calinski_harabasz"],
                    "adjusted_rand_index": ext_metrics["adjusted_rand_index"],
                    "normalized_mutual_info": ext_metrics["normalized_mutual_info"],
                    "seed_stability_ari": stab["mean_seed_stability_ari"],
                    "noise_ratio": metrics["noise_ratio"],
                }
            )

    benchmark_df = pd.DataFrame(benchmark_rows)

    # Save to data/processed/benchmark.csv
    proc_dir = Path(config["paths"]["processed_dir"])
    proc_dir.mkdir(parents=True, exist_ok=True)
    benchmark_df.to_csv(config["paths"]["benchmark_csv"], index=False)

    return benchmark_df


if __name__ == "__main__":
    cfg = load_config()
    feat_df = pd.read_csv(cfg["paths"]["features_csv"])
    gt_df = pd.read_csv(cfg["paths"]["ground_truth_csv"])

    feat_cols = [c for c in feat_df.columns if c not in ["platform_id", "platform_name"]]
    X = feat_df[feat_cols].to_numpy(dtype=float)

    bench = run_ablation_benchmarking(X, gt_df, cfg)
    print("Benchmark summary (top 10 by ARI):")
    print(
        bench.sort_values(by="adjusted_rand_index", ascending=False)[
            ["representation", "algorithm", "silhouette", "adjusted_rand_index", "normalized_mutual_info", "seed_stability_ari"]
        ].head(10)
    )
